
import itk
import torch
from itk import RTK as rtk
from deepinv.physics.forward import LinearPhysics, adjoint_function

import matplotlib.pyplot as plt

if "cuda_image_from_cuda_array" not in itk.__dir__():

    raise Exception("Cuda-Array-Interface is not available in the environment. Please update your ITK version.")

#    .. warning::
#        The forward and backprojectors are not strictly matched adjoints when using voxel based backprojections.


class TomographyWithRTK(LinearPhysics):
    r"""Computed Tomography operator with ITK-RTK CUDA backend.

    This operator implements a ray transform :math:`A` using the RTK (Reconstruction Toolkit) GPU-accelerated forward and backprojectors.

    Mathematically, it computes line integrals of an object :math:`x` along X-ray paths:

    .. math::
        y = Ax

    where :math:`y` represents projection data.

    Supported geometries:

    * ``fanbeam`` (2D-like configuration embedded in 3D)
    * ``conebeam`` (3D circular trajectory)

    The adjoint is implemented using RTK's ray-cast backprojection.
    The pseudo-inverse can be approximated using FDK reconstruction.



    .. note::

        This implementation requires:
        - ITK compiled with CUDA support
        - RTK module enabled
        - Cuda-Array-Interface support in ITK
    """

    def __init__(
        self,
        geometry:any,
        detector_information: dict[str, list],
        imagesource_information: dict[str, list],
        mode:str,
        verbose:bool=False,
        normalize:bool=False,
        stepsize:bool=None,
        *args,
        **kwargs,
    ):
        """
        :param geometry: RTK geometry object defining source/detector trajectory.
        :param detector_information: Dictionary describing detector grid
            (size, spacing, origin, offset).
        :param imagesource_information: Dictionary describing reconstruction
            volume grid (size, spacing, origin).
        :param str mode: Either ``"fanbeam"`` or ``"conebeam"``.
        :param bool verbose: If True, print geometry configuration.
        :param bool normalize: If True, normalize operator to unit norm.
        :param float stepsize: Ray marching step size.
        """
            
        super().__init__(*args, **kwargs)
            
        self._NB_STACK = 2
        self._NB_STACK_PROJ = 1
        self._CUDA_IMAGE_TYPE = itk.CudaImage[itk.F, 3]

        self.geometry = geometry
        self.projectionOffY = detector_information["offset"]
        self.normalize = normalize
        self.mode = mode
        
        self.detector_information = detector_information
        self.imagesource_information = imagesource_information
        
        if mode not in ("fanbeam", "conebeam"):
            raise ValueError(f"mode {mode!r} unrecognized (expected 'fanbeam' or 'conebeam')")

        self._validate_info(mode, self.detector_information, "detector_information")
        self._validate_info(mode, self.imagesource_information, "imagesource_information")


        # ---------------------------------------------------------
        # FANBEAM CONFIGURATION (2D embedded in 3D volume)
        # ---------------------------------------------------------
        if mode == 'fanbeam':
            
            self.detector_information["spacing"] = [
                self.detector_information["spacing"][0],
                self.detector_information["spacing"][0],
                self.detector_information["spacing"][1] 
            ]
            self.detector_information["size"] = [
                self.detector_information["size"][0],
                self._NB_STACK_PROJ,
                self.detector_information["size"][1] 
            ]
            self.detector_information["origin"] = [
                self.detector_information["origin"][0],
                -(self.detector_information["spacing"][1] * self._NB_STACK_PROJ/2 + self.projectionOffY) + self.detector_information["spacing"][1]/2 ,
                self.detector_information["origin"][1] 
            ]

            
            self.imagesource_information["spacing"] = [
                self.imagesource_information["spacing"][0],
                self.imagesource_information["spacing"][0],
                self.imagesource_information["spacing"][1] 
            ]
            self.imagesource_information["size"] = [
                self.imagesource_information["size"][0],
                self._NB_STACK,
                self.imagesource_information["size"][1] 
            ]
    
            self.imagesource_information["origin"] = [
                self.imagesource_information["origin"][0],
                -self.imagesource_information["spacing"][1] * self._NB_STACK / 2 + self.imagesource_information["spacing"][1]/2,
                self.imagesource_information["origin"][1] 
            ]

        if stepsize is None:
            self.stepsize = self.imagesource_information["spacing"][1] 
        else:
            self.stepsize = stepsize

        if normalize:
            self.norm_mat = self.compute_norm(torch.randn((1, 1, *self.imagesource_information["size"]), device=device)).sqrt()
        else:
            self.norm_mat = None

        if verbose:
            print("Detector Informations : ")
            print(self.detector_information["size"]) 
            print(self.detector_information["origin"]) 
            print(self.detector_information["spacing"]) 

            print("Source Informations : ")
            print(self.imagesource_information["size"]) 
            print(self.imagesource_information["origin"]) 
            print(self.imagesource_information["spacing"]) 
            
    def _validate_info(self, mode: str, info: dict[str, int | float], name: str) -> None:
        """ Validate that all array-like values in ``info`` have the expected length for the given mode.
        
        :param mode: Either ``"fanbeam"`` or ``"conebeam"``.
        :param info: Dictionary describing a grid (size, spacing, origin) in the image space or in the observation space.
        :param name: Name for error throwing.
        """
        expected_len = 2 if mode == "fanbeam" else 3
        
        for key, element in info.items():
            # If the element is not array like, ignore
            if not hasattr(element, "__len__"):
                continue
                
            actual_len = len(element)
            if actual_len != expected_len:
                raise ValueError(
                    f"Expected element length {required_len} for mode {mode!r}, "
                    f"got {actual_len} for key {key!r} in {name}"
                )
            
    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward projection.

        :param torch.Tensor x: input of shape [B,C,...,H,W]
        :return: projection of shape [B,C,...,A,N]
        """
        
        x_stacked = x.squeeze(0).squeeze(0)

        if self.mode == 'fanbeam':
            x_stacked = torch.stack([x_stacked.clone()] * self._NB_STACK, dim=1).to("cuda:0") # stack 4 slices of x

        imageSource_cuda = itk.cuda_image_from_cuda_array(x_stacked)
        imageSource_cuda.SetOrigin(self.imagesource_information["origin"])
        imageSource_cuda.SetSpacing(self.imagesource_information["spacing"])
        
        fp_source = rtk.ConstantImageSource[self._CUDA_IMAGE_TYPE].New()
        fp_source.SetSize(self.detector_information["size"])
        fp_source.SetOrigin(self.detector_information["origin"]) 
        fp_source.SetSpacing(self.detector_information["spacing"])
        fp_source.SetConstant(0.0)
        fp_source.Update()
    
    
        # 1. Forward projection: Ax
        forward_projector = rtk.CudaForwardProjectionImageFilter[self._CUDA_IMAGE_TYPE].New()
        forward_projector.SetGeometry(self.geometry)
        forward_projector.SetInput(fp_source.GetOutput())
        forward_projector.SetInput(1, imageSource_cuda)
        forward_projector.SetStepSize(self.stepsize)
        forward_projector.Update()
        Ax = forward_projector.GetOutput()
        Ax.DisconnectPipeline()


        projections = torch.as_tensor(Ax, device=x.device).clone()

        if self.mode == 'fanbeam':
            projections = projections[:, 0, :]

        if self.normalize:
            projections /= self.norm_mat            
        
        return projections.unsqueeze(0).unsqueeze(0)
        
    
    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Approximation of the adjoint.

        :param torch.Tensor y: input of shape [B,C,...,A,N]
        :return: scaled back-projection of shape [B,C,...,H,W]
        """
        
        y_stacked = y.squeeze(0).squeeze(0)

        if self.mode == 'fanbeam':
            y_stacked = torch.stack([y_stacked.clone()] * self._NB_STACK_PROJ, dim=1) # stack 4 slices of x

        projection_cuda = itk.cuda_image_from_cuda_array(y_stacked)
        projection_cuda.SetOrigin(self.detector_information["origin"])
        projection_cuda.SetSpacing(self.detector_information["spacing"])
        
        fp_source = rtk.ConstantImageSource[self._CUDA_IMAGE_TYPE].New()

        fp_source.SetSize(self.imagesource_information["size"])
        fp_source.SetOrigin(self.imagesource_information["origin"]) 
        fp_source.SetSpacing(self.imagesource_information["spacing"])
        fp_source.SetConstant(0.0)
        fp_source.Update()
            
        # 3. Backprojection: A^T (Ax - p)
        back_projector = rtk.CudaRayCastBackProjectionImageFilter.New()
        back_projector.SetGeometry(self.geometry)
        back_projector.SetInput(0, fp_source.GetOutput())
        back_projector.SetInput(1, projection_cuda)
        back_projector.SetStepSize(self.stepsize)
        back_projector.Update()
        Atx = back_projector.GetOutput()
        Atx.DisconnectPipeline()

        backproj = torch.as_tensor(Atx, device=y.device).clone()
        #reconstruction = reconstruction_whole[:, 0, :] * 2
        
        if self.mode == 'fanbeam':
            backproj = backproj.sum(dim=1)
            
        if self.normalize:
            backproj /= self.norm_mat

        return backproj.unsqueeze(0).unsqueeze(0)

    
    def fbp(self, y: torch.Tensor, parker_angle:float=0, pad:float=0, **kwargs) -> torch.Tensor:
        """
        USING FDK [XXX]
        """
        y_stacked = y.squeeze(0).squeeze(0)
        
        if self.mode == 'fanbeam':
            y_stacked = torch.stack([y_stacked.clone()] * self._NB_STACK_PROJ, dim=1) # stack 4 slices of x

        projection_cuda = itk.cuda_image_from_cuda_array(y_stacked)
        projection_cuda.SetOrigin(self.detector_information["origin"])
        projection_cuda.SetSpacing(self.detector_information["spacing"])
        
        fp_source = rtk.ConstantImageSource[self._CUDA_IMAGE_TYPE].New()

        fp_source.SetSize(self.imagesource_information["size"])
        fp_source.SetOrigin(self.imagesource_information["origin"]) 
        fp_source.SetSpacing(self.imagesource_information["spacing"])
        fp_source.SetConstant(0.0)
        fp_source.Update()                
        
        # FDK reconstruction
        parker = rtk.CudaParkerShortScanImageFilter.New(Geometry=self.geometry)
        parker.SetInput( projection_cuda )
        parker.SetAngularGapThreshold(parker_angle)

        feldkamp = rtk.CudaFDKConeBeamReconstructionFilter.New()
        feldkamp.SetInput(0, fp_source.GetOutput())
        feldkamp.SetInput(1, parker.GetOutput())
        feldkamp.SetGeometry(self.geometry)
        feldkamp.GetRampFilter().SetTruncationCorrection(pad)
        feldkamp.GetRampFilter().SetHannCutFrequency(0.0)

        feldkamp.Update()
        
        itk_reco = feldkamp.GetOutput()
        itk_reco.DisconnectPipeline()

        reco = torch.as_tensor(itk_reco, device=y.device).clone()
        
        if self.mode == 'fanbeam':
            reco = reco.sum(dim=1)
            
        return reco.unsqueeze(0).unsqueeze(0)