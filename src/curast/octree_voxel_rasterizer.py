import torch

from curast import cuda

class OctreeVoxelRasterizer:
    @staticmethod
    def rasterize(
        positions,
        colors_precomp,
        densities,
        depths,
        aabb,
        raster_settings,
    ):
        args = (
            raster_settings.bg,
            positions,
            colors_precomp,
            densities,
            depths,
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            aabb,
            raster_settings.with_distloss
        )

        num_rendered, color, depth, alpha, distloss, geom_buffer, binning_buffer, img_buffer = cuda.rasterize_cuda_voxels(*args)

        return color, depth, alpha, distloss
        