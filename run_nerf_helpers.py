import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm, trange

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2abs = lambda x, y : torch.mean(torch.abs(x - y))
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    
def init_weights_zeros_(m):
    if type(m) == nn.Linear:
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)
    
    
    
class MLP_IOR(nn.Module):
    def __init__(self, D=8, W=64, input_ch=3 + 3*2*6, output_ch=1, skips=[4],is_index=False):
        """ 
        """
        super(MLP_IOR, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.is_index = is_index
        self.mlp_ior = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
       
        self.mlp_ior_end = nn.Linear(W, output_ch)
        self.softplus = nn.Softplus(beta=5)
        # self.IOR = nn.Parameter(0.5*torch.ones(1))
    def forward(self, x):
        input_pts =  x
        h = input_pts
        for i, l in enumerate(self.mlp_ior):
            h = self.mlp_ior[i](h)
            h = self.softplus(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

       
        outputs = self.mlp_ior_end(h)
        shape_out = self.softplus(outputs)
                
        return shape_out 
# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

dtype_long = torch.cuda.LongTensor

def trilinear_interpolation(inputs,xq,scene_bound):
    
    res = inputs.shape[0]
    _min = scene_bound[0,:]
    _max = scene_bound[1,:]
    xq = (res-1)*(xq-_min)/(_max-_min)
    xq = xq.clip(0,res-1)
    x = xq[:,0]
    y = xq[:,1]
    z = xq[:,2]
    
    x_0 = torch.clamp(torch.floor(x).type(dtype_long),0, res-1)
    x_1 = torch.clamp(x_0 + 1, 0, res-1)
    
    y_0 = torch.clamp(torch.floor(y).type(dtype_long),0, res-1)
    y_1 = torch.clamp(y_0 + 1, 0, res-1)
    
    z_0 = torch.clamp(torch.floor(z).type(dtype_long),0, res-1)
    z_1 = torch.clamp(z_0 + 1, 0, res-1)
 
    
    u, v, w = x-x_0, y-y_0, z-z_0
    u = u[:,None]
    v = v[:,None]
    w = w[:,None]
    
    c_000 = inputs[x_0,y_0,z_0]
    c_001 = inputs[x_0,y_0,z_1]
    c_010 = inputs[x_0,y_1,z_0]
    c_011 = inputs[x_0,y_1,z_1]
    c_100 = inputs[x_1,y_0,z_0]
    c_101 = inputs[x_1,y_0,z_1]
    c_110 = inputs[x_1,y_1,z_0]
    c_111 = inputs[x_1,y_1,z_1]
    # print(c_111.shape)
    # print(u.shape)

    c_xyz = (1.0-u)*(1.0-v)*(1.0-w)*c_000 + \
            (1.0-u)*(1.0-v)*(w)*c_001 + \
            (1.0-u)*(v)*(1.0-w)*c_010 + \
            (1.0-u)*(v)*(w)*c_011 + \
            (u)*(1.0-v)*(1.0-w)*c_100 + \
            (u)*(1.0-v)*(w)*c_101 + \
            (u)*(v)*(1.0-w)*c_110 + \
            (u)*(v)*(w)*c_111  
            


    return c_xyz


    
def lowpass_3d(res,sigma):
    
    res = res-1
    xx, yy,zz = torch.meshgrid(torch.linspace(0, res, res+1), torch.linspace(0, res, res+1), torch.linspace(0, res, res+1))  
    xx = xx/res
    yy = yy/res
    zz = zz/res
    dist = (torch.square(xx-0.5)+torch.square(yy-0.5)+torch.square(zz-0.5))
   
    return torch.exp(-dist/(2*sigma**2))


def voxel_lowpass_filtering(input_voxel,filter_3d):
    
    fftn_grid = torch.fft.fftshift(torch.fft.fftn(input_voxel,input_voxel.shape,norm='forward'))
    filtered_grid = torch.fft.ifftn(torch.fft.ifftshift(fftn_grid*filter_3d[...,None].to(input_voxel)),norm='forward')
    
    
    return filtered_grid.real


eps = 1e-6
def normalizing(vec):
    
    return vec/(torch.linalg.norm(vec, dim=-1, keepdim=True)+eps)



def get_scene_bound(near,far,H,W,K,poses,min_=.25,max_=.25):
        
        
    N_samples = 2
    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    
    z_vals = z_vals.expand([H,W, N_samples])
    scene_bound_min = []
    scene_bound_max = []
    
    for cams in tqdm(range(len(poses))):
        rays_o, rays_d = get_rays(H, W, K, poses[cams])
        pts = rays_o[...,None,:] + (rays_d[...,None,:]) * z_vals[...,:,None]
        pts = pts.reshape(-1,3)
        max_pts = torch.quantile((pts),1.-max_,axis=0)
        min_pts = torch.quantile((pts),min_,axis=0)
        scene_bound_min.append(min_pts)
        scene_bound_max.append(max_pts)
    
    
    scene_bound_min = torch.stack(scene_bound_min,0)
    scene_bound_max = torch.stack(scene_bound_max,0)
    
    max_pts = torch.quantile(scene_bound_max,1.-max_,axis=0)
    min_pts = torch.quantile(scene_bound_min,min_,axis=0)
        
    return torch.stack([min_pts,max_pts],0)


def get_bb_weights(pts,bounding_box_val,beta=200):
    
    center  = bounding_box_val[0] 
    rad = bounding_box_val[1]

    x_dist = torch.abs(pts[...,0:1] - torch.tensor(center[0]).to(pts))
    y_dist = torch.abs(pts[...,1:2] - torch.tensor(center[1]).to(pts))
    z_dist = torch.abs(pts[...,2:3] - torch.tensor(center[2]).to(pts))

    weights = torch.sigmoid(beta*(rad[0]-x_dist))*torch.sigmoid(beta*(rad[1]-y_dist))*torch.sigmoid(beta*(rad[2]-z_dist))

    return 1.0 - weights




def get_voxel_grid(voxel_res,scene_bound,poses,query_fn,nerf_model,masking=False,bb_vals=None,flag=False):
    
    
    min_bound,max_bound = scene_bound.cpu().numpy()

    
    x  = np.arange(voxel_res)
    grid_x,grid_y,grid_z = np.meshgrid(x,x,x)
    
    pts_ = np.stack((grid_y,grid_x,grid_z),-1)
    pts_ = pts_.reshape(-1,3)
    pts_ = pts_/(voxel_res-1)
    pts_ = (pts_)*(max_bound-min_bound) + min_bound
    
    raw_avg = 0
    
    for cam in tqdm(range(len(poses))):
        
        raw_stack = []
        pose = poses[cam,:,-1]

        batch = 128*128*128
        for ii in range(0,pts_.shape[0],batch):
            with torch.no_grad():
                
                new_pts_ = torch.tensor(pts_[ii:ii+batch,:]).to(poses).to(torch.float32)
                viewdirs =  new_pts_ -  pose  
            
                viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
                
                raw = query_fn(new_pts_, viewdirs, nerf_model)
                raw[...,0:3] = torch.sigmoid(raw[...,0:3])
                raw[...,3:4] = F.relu(raw[...,3:4])
                raw_stack.append(raw)
                
        raw_stack = torch.cat(raw_stack,0)   

            
        raw_avg += raw_stack  

    raw_avg = raw_avg/len(poses)
    return raw_avg.reshape(voxel_res,voxel_res,voxel_res,4)





def raw2outputs(raw, dists):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)


    
    alpha = raw2alpha(raw[...,3] , dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    return weights
