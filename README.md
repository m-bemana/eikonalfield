# Eikonal Fields for Refractive Novel-View Synthesis  [[Project Page]](https://eikonalfield.mpi-inf.mpg.de) 


<p align="center">
<img src = "https://user-images.githubusercontent.com/69102582/177169284-f020de1a-e7de-45da-9768-c956ec225b0a.gif" width=100%>
</p>

## Installation

```
conda create -n eikonalfield python=3.8
conda activate eikonalfield
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (Click to expand)</summary>
 
### Dependencies 
* torch>=1.8
* torchvision>=0.9.1      
* matplotlib
* imageio
* imageio-ffmpeg
* configargparse
* tqdm
* opencv-python                  
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq)             

</details>

## Dataset

<img src = "https://user-images.githubusercontent.com/69102582/179800272-fd37f884-0db0-4f09-928e-1853ac593e53.png" width=100%>


* [``Ball``](https://eikonalfield.mpi-inf.mpg.de//datasets/Ball.zip)  
* [``Glass``](https://eikonalfield.mpi-inf.mpg.de//datasets/Glass.zip)  
* [``Pen``](https://eikonalfield.mpi-inf.mpg.de//datasets/Pen.zip) 
* [``WineGlass``](https://eikonalfield.mpi-inf.mpg.de//datasets/WineGlass.zip)

Each dataset contains the captured images and a short video of the scene.
In the `captured images` folder, we provide the images with the original 4K resolution and a smaller resolution with the estimated camera poses using the COLMAP and [LLFF code](https://github.com/fyusion/llff). In the `captured video` folder, the video frames with their estimated camera poses are provided.  


## Training

* __Step 0__: Finding the camera poses ``poses_bounds.npy``with the instruction given [here](https://github.com/bmild/nerf#generating-poses-for-your-own-scenes) 
 
   (For our dataset the camera parameters are already provided!)
* __Step 1__: Estimating the radiance field for the entire scene by running ``run_nerf.py`` (The code is borrowed from [``nerf-pytorch``](https://github.com/yenchenlin/nerf-pytorch))
  ```
    python run_nerf.py --config  configs/Glass.txt 
  ```
     The config files for the scenes in our dataset are located in the `configs` folder. 

* __Step 2__: Finding the 3D bounding box containing the transparent object using ``find_bounding_box.py``.
  ```
    python find_bounding_box.py --config configs/Glass.txt
  ```

   <details>
   <summary>  (Click to expand) In this step, 1/10 of the training images are displayed in order to mark a few points at the extent of the transparent object. </summary>   

    <img src = "https://user-images.githubusercontent.com/69102582/178770875-71b40783-bfa1-454a-a431-4e06676c9aaf.png" width=100%>


   </details>   

* __Step 3__: Learning the index of refraction (IoR) field with ``run_ior.py``
  ```
    python run_ior.py --config configs/Glass.txt --N_rand 32000 --N_samples 128
  ```
* __Step 4__: Learning the radiance field for the object inside the transparent object using ``run_nerf_inside.py``
  ``` 
  python run_nerf_inside.py --config configs/Glass.txt  --N_samples 512
  ```
  (Please note that for the Ball scene we skipped this step)  

## Rendering

Please run the ``render_model.py`` with different modes to render the learned models at each training step.  

```
  python render_model.py --config configs/Glass.txt  --N_samples 512  --mode  1  --render_video
  ```
  
  The rendering options are:

```
           --mode                  # use 0 to render the output of step 1 (Original NeRF)
                                   # use 1 to render the output of step 3 (Learned IoR)
                                   # use 2 to render the output of step 4 (Complete model with the inside NeRF) 
           --render_test           # rendering the test set images
           --render_video          # rendering a video from a precomputed path
           --render_from_path      # rendering a video from a specified path                      
```
## Models

Please find below our results and pre-trained  model for each scene:
* [``Ball``](https://eikonalfield.mpi-inf.mpg.de//results/Ball.zip)  
* [``Glass``](https://eikonalfield.mpi-inf.mpg.de//results/Glass.zip)  
* [``Pen``](https://eikonalfield.mpi-inf.mpg.de//results/Pen.zip) 
* [``WineGlass``](https://eikonalfield.mpi-inf.mpg.de//results/WineGlass.zip)

Each scene contains the following folders:

* ``model_weights``   --> the pre-trained model
* ``bounding_box``   ---> the parameters of the bounding box 
* ``masked_regions`` ---> the masked images identifying the regions crossing the bounding box in each view
* ``rendered_from_a_path``  ---> the rendered video result along the camera trajectory of the real video capture 


## Details
### Capturing
Our method works with a general capturing setup and does not require any calibration pattern or a specific setup. We spherically capture the scene and get close enough to the transparent object to properly sample the transparent object.


### Bounding Box
Our bounding box (BB) is a rectangular cuboid parameterized by its center $c = (c_x,c_y,c_z)$ and the distances from the center to a face in each dimension $d=(d_x,d_y,d_z)$. 
For a 3D point $(x,y,z)$ in the space, the bounding box is analytically expressed as follows:

 $$ 𝐵𝐵(𝒙,𝒚,𝒛)= 1 - 𝑆(𝛽*(d_𝑥−|𝒙−𝑐_𝑥 |)). 𝑆(𝛽*(d_𝑦−|𝒚−𝑐_𝑦 |)) . 𝑆(𝛽*(d_𝑧−|𝒛−𝑐_𝑧 |)) $$


 where 
$𝑆(𝑥)=\frac{1}{1+𝑒^{−𝑥}}$   is the sigmoid function and $\beta$ is the steepness coefficient. We use $\beta=200$ in our experiments. Using this function, a point inside the box gets a zero value and the points outside get a value close to one. 

### Voxel grid
In our IoR optimizations we first need to smooth the learned radiance field; however, explicitly smoothing an MLP-based radiance field is not straightforward, 
we instead fit a uniform 3D grid to the learned radiance field. We then band-limit the grid in the Fourier domain using a Gaussian blur kernel to obtain the coarse-to-fine radiance field.
Note we fit the voxel grid to the NeRF coarse model rather than the fine one to avoid aliasing, and for the spherical captures, we limit the scene far bound to 2.5.

### IoR optimization
Since we have a complete volume rendering model in the form of ODEs, we use a differentiable ODE solvers package provided by the [Neural ODE](https://github.com/rtqichen/torchdiffeq) to backpropagate through the ODEs. Moreover, using this package our training proceeds in a memory independent of the step count which allows the processing of more rays (as large as 32k rays) in each iteration.  

### IoR model
When using differentiable ODE solvers, we found it very important to use a smooth non-linear activation such as Softplus in our IoR MLP model otherwise the optimization becomes unstable.
 

### Rendering 
Since we could not utilize a hierarchical sampling in our volume rendering with ODE formulation, we consider
512 steps along the ray to properly sample both interior and exterior radiance fields.



## Citation

    @inproceedings{bemana2022eikonal,
        title={Eikonal Fields for Refractive Novel-View Synthesis},
        author={Bemana, Mojtaba and Myszkowski, Karol and Revall Frisvad, Jeppe and Seidel, Hans-Peter and Ritschel, Tobias},
        booktitle={Special Interest Group on Computer Graphics and Interactive Techniques Conference Proceedings},
        pages={1--9},
        year={2022}

## Contact
mbemana@mpi-inf.mpg.de
