
"""
2-3D CBCT volume reconstruction using the ITK-RTK backend and Total-Variation (TV) prior
========================================================================================

This example shows 3D cone-beam CT (CBCT) reconstruction using the Reconstruction Toolkit (RTK) library.

We generate a Shepp–Logan phantom, simulate its projection using the RTK CUDA projector operator, and add noise to the measurements. The reconstruction is thus performed using a Proximal Gradient Descent (PGD) algorithm with a Total Variation (TV) prior, and the result is compared against the Feldkamp–Davis–Kress (FDK) reconstruction.

We do this through the `TomographyWithRTK operator`, which wraps the RTK <https://docs.openrtk.org/en/latest/>_ library. This class is defined at this end of this example.

This example requires ITK and RTK to be installed in your current environment. They can be installed using: `pip install itk itk-rtk`

"""
import itk
from itk import RTK as rtk
import torch
import deepinv as dinv
from pathlib import Path

from deepinv.optim.data_fidelity import L2
from deepinv.optim import PGD
from deepinv.utils.plotting import plot, plot_curves
from deepinv.physics import GaussianNoise

import matplotlib.pyplot as plt

# %%
# Definition of the tomography operator class
# ----------------------------------------------------------------------------------------
#
# We define this class in another file to be imported into the main script. This class handles the projection and backprojection operators through RTK.

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
            self.norm_mat = self.compute_norm(torch.randn(1, 1, self.imagesource_information["size"][0], self.imagesource_information["size"][1], device=device)).sqrt()
            # Change this <<<<<<<<<<<<<<<<
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

        expected_len = 2 if mode == "fanbeam" else 3
        
        for key, element in info.items():
            if not hasattr(element, "__len__"):
                continue
                
            actual_len = len(element)
            if actual_len != expected_len:
                raise ValueError(
                    f"Expected element length {required_len} for mode {mode!r}, "
                    f"got {actual_len} for key {key!r} in {name}"
                )

    @classmethod
    def from_yaml(cls, config_file, verbose=False, *args, **kwargs):
        volume_grid, projection_grid, geometry, stepsize = read_recon_config(config_file)

        return cls(
            geometry=geometry, 
            detector_information=projection_grid.to_dict('2d'), 
            imagesource_information=volume_grid.to_dict('2d'), 
            verbose=verbose, stepsize=stepsize, *args, **kwargs
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


# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#
BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"


# %%
# Generation of the Shepp-Logan phantom
# -----------------------------------------
# Here, we genetare the 3D phantom through the RTK library. And convert it from an ITK image to a Torch tensor.
# Note that the phantom value es are from 0 to 2 rather than the normalized 0 to 1. And its pixel value dyanmica are much different [0, 1.02, 1.04, 1.06, 2] to make it closeer to real CBCT acqustion of the human body. (soft tissue and bones)

ImageType = itk.Image[itk.F, 3]
slice_of_interest = 24 # Use for visualization

source = rtk.constant_image_source(
    size=[64, 64, 64], spacing=[4.0] * 3, ttype=[ImageType], origin=[-128.0, -128.0, -128]
)

shepploganFilter = rtk.DrawSheppLoganFilter.New()
shepploganFilter.SetInput(source)
shepploganFilter.Update()

# Convert to pytorch tensor
itk_gpu_image = itk.cuda_image_from_image(shepploganFilter.GetOutput())
itk_gpu_tensor = torch.tensor(itk_gpu_image).unsqueeze(0).unsqueeze(0)

# Display slice of the volume
plt.imshow((itk_gpu_tensor.cpu()[0, 0, :, slice_of_interest, :]), vmin=0.99, vmax=1.04, cmap="gray")
plt.colorbar()
plt.show()


# %%
# Definition of forward operator and noise model
# -----------------------------------------------
# First we define the geometry and then dfine the detector and image source corrdinate information
# For the noise level we suppose the noise following a gaussian dirbution.
#

# Setup the geometry 
numberOfProjections = 600
angularArc = 360.0
sid = 300
sdd = 500 
geometry = rtk.ThreeDCircularProjectionGeometry.New()
for i in range(0, numberOfProjections):
    angle = i * angularArc / numberOfProjections
    geometry.AddProjection(sid, sdd, angle)

detector_information = {
    "spacing": [1, 1, 1],
    "size": [100, 100, 600],
    "origin": [-50, -50, 0],
    "offset": 0,
}


imagesource_information = {
    "spacing": [1, 1, 1],
    "size": [64, 64, 64],
    "origin": [-32.0, -32.0, -32],
}

# Instantiation of the operator
noise_level = 3e-1
physics = TomographyWithRTK(
    geometry=geometry, 
    detector_information=detector_information, 
    imagesource_information=imagesource_information, 
    verbose=True, 
    normalize=False, 
    noise_model=GaussianNoise(sigma=noise_level),
    mode='conebeam',
    stepsize=0.15
)

# Application of the operator and computing the pseudo inverse using the FDK algorithm
observation = physics(itk_gpu_tensor)
fdk = physics.fbp(observation)

# %%
# Set up the optimization algorithm to solve the inverse problem.
# --------------------------------------------------------------------------------
# The problem we want to minimize is the following:
#
# .. math::
#
#     \begin{equation*}
#     \underset{x}{\operatorname{min}} \,\, \frac{1}{2} \|Ax-y\|_2^2 + \lambda \|Dx\|_{1,2}(x),
#     \end{equation*}
#
#
# where :math:`1/2 \|A(x)-y\|_2^2` is the a data-fidelity term, :math:`\lambda \|Dx\|_{2,1}(x)` is the total variation (TV)
# norm of the image :math:`x`, and :math:`\lambda>0` is a regularisation parameters.
#
# We use a Proximal Gradient Descent (PGD) algorithm to solve the inverse problem.


# Select the data fidelity term
data_fidelity = L2()
prior = dinv.optim.prior.TVPrior(n_it_max=20)

# Logging parameters
verbose = True
plot_convergence_metrics = (
    True  # compute performance and convergence metrics along the algorithm.
)

# Algorithm parameters
print("Calculating the operator norm this may take some time...")
scaling = 1/ physics.compute_sqnorm(torch.randn_like(itk_gpu_tensor), max_iter=100, tol=1e0).item()
print(scaling)

stepsize = 0.99 * scaling
lamb = 20  # TV regularisation parameter
max_iter = 600 # 300
early_stop = True

# Instantiate the algorithm class to solve the problem.
model = PGD(
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    verbose=verbose,
    stepsize=stepsize,
    lambda_reg=lamb,
    custom_init=lambda observation, physics: scaling * physics.A_adjoint(observation),  # initialize the optimization with backprojection
)


# %%
# Evaluate the model on the problem and plot the results.
# --------------------------------------------------------------------
#
# The model returns the output and the metrics computed along the iterations.
# The PSNR is computed w.r.t the ground truth image in ``test_imgs``.

# run the model on the problem.
x_model, metrics = model(
    observation, physics, x_gt=itk_gpu_tensor, compute_metrics=True
)  # reconstruction with PGD algorithm

# compute PSNR
print(
    f"Filtered Back-Projection PSNR: {dinv.metric.PSNR(max_pixel=itk_gpu_tensor.max().item())(itk_gpu_tensor.unsqueeze(0), fdk).item():.2f} dB"
)
print(
    f"PGD reconstruction PSNR: {dinv.metric.PSNR(max_pixel=itk_gpu_tensor.max())(itk_gpu_tensor.unsqueeze(0), x_model).item():.2f} dB"
)

imgs = [itk_gpu_tensor[0, :, :, slice_of_interest, :], fdk[0, :, :, slice_of_interest, :], x_model[0, :, :, slice_of_interest, :]]
plot(
    imgs,
    titles=["GT", "Filtered Back-Projection", "Recons."],
    save_dir=RESULTS_DIR,
    vmin=0.99,
    vmax=1.04,
    rescale_mode='clip'
)

# plot convergence curves
if plot_convergence_metrics:
    plot_curves(metrics, save_dir=RESULTS_DIR)

plot(
    observation[0, :, 50, :, :], 
    titles=[f"Noisy sinogram"],
    vmax=55,
    save_dir=RESULTS_DIR,
    rescale_mode='clip'
)