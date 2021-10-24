from pathlib import Path
import logging

app_name = 'Qiber3D'
"""Name of the app"""
version_number = (0, 7, 0)
"""tuple(int): version number"""
version = f'{version_number[0]}.{version_number[1]}.{version_number[2]}'
"""str: version"""
app_author = 'Anna Jaeschke; Hagen Eckert'
"""str: Authors"""
url = 'https://github.com/theia-dev/Qiber3D'
"""str: git url"""
log_level = logging.INFO
"""int: default logging level. Use :meth:`Qiber3D.helper.change_log_level` to change it on the fly"""
core_count = 0
"""int: CPU core count (0 = autodetect)"""
base_dir = Path(__file__).absolute().parent


# settings for the rendering engine
class render:
    """
    Settings for rendering images and animations
    """
    ffmpeg_path = 'ffmpeg'
    """str: path to local installation of ffmpeg"""
    rgba = True
    """bool: allow transparency for images"""
    background = (0.0, 0.0, 0.0)
    """tuple(float): background color as RGB values (0-1)"""
    animation_height = 720
    """int: vertical resolution for animations"""
    image_resolution = 3840
    """int: horizontal resolution for images"""
    color = (0.9, 0.2, 0.2)
    """tuple(floats): standard foreground color"""
    notebook = False
    """bool: switch to adapt render pipeline for jupyter notebooks (automatically set)"""


# settings for figure
class figure:
    """
    Settings for matplotlib based figures
    """
    grid_lw = 0.25
    """float: grid line width for directions figure"""
    grid_color = (0.2, 0.2, 0.2)
    """tuple(float): grid rgb color tuple for directions figure"""
    format = '.pdf'
    """str: default figure export format"""
    dpi = 300
    """int: figure resolution"""


class extract:
    """
    Parameter to extract a :class:`Qiber3D.Network` from an image stack
    """
    nd2_channel_name = 0
    """str or int: channel name or index when importing from nd2 files"""
    save_steps = True
    """bool: save extraction steps compressed in memory"""
    low_memory = False
    """bool: reduce the memory footprint by using less precision when possible"""
    voxel_size = None
    """list(float): size of a voxel in each axis"""
    invert = False
    """bool: invert the input image (extracted structures should be high and background low)"""
    use_teasar = False
    """bool: if ``True`` use the `kimimaro <https://github.com/seung-lab/kimimaro>`_ 
    TEASAR implementation for reconstruction"""

    class z_drop:
        """Z-Drop - Intensity attenuation correction"""
        apply = True
        """bool: apply the intensity attenuation correction to the image"""

    class isotropic_resampling:
        """Rescaling to cubic voxels"""
        target = 'z'
        """string: select if the ``z`` resolution or the ``xy`` resolution should be adjust to form cubic voxels"""

    class median:
        """Median filter - despeckle"""
        apply = True
        """bool: apply the median filter to the image"""
        size = 3
        """(int or array): size of the neighborhood cuboid, if an int is given the all three axis have the same length"""
        footprint = None
        """array: If set, this 3D binary array is used instead of the ``size`` parameter to describe the neighborhood"""

    class binary:
        """Binarization"""
        threshold = 'Otsu'
        """(float or string): binarization threshold in percent, or name of the following auto threshold methods -
        ``Otsu``, ``Isodata``, ``Li``, ``Mean``, ``Minimum``, ``Triangle``, Yen (`examples <https://scikit-image.org/docs/0.18.x/auto_examples/segmentation/plot_thresholding.html>`_)"""

    class morph:
        """Morphological dilation and erosion"""
        apply = True
        """bool: apply the dilation and erosion to the image"""
        iterations = 5
        """int: number of iterations"""
        remove_vol = 100
        """float: remove islands with volume smaller than volume smaller than ``remove_vol`` - in (voxel_size units)^3"""

    class smooth:
        """Gaussian filter"""
        apply = True
        """bool: apply a gaussian filter and erosion to the image"""
        sigma = 2.0
        """float: standard deviation for Gaussian kernel in voxel"""
        truncate = 2.0
        """float: truncate the filter after this many standard deviations"""

    class thinning:
        """Thinning based reconstruction"""
        voxel_per_point = 10
        """float: distance between interpolated points (every segment will have at least five points)"""
        sliver_threshold = 6
        """int: segments with less than ``sliver_threshold`` voxels will be treated specially (minimum of 6)"""
        distance_voxel_overlap = 15
        """int: Overlap in voxel for the low memory euclidean distance transformation"""

    class teasar:
        """TEASER reconstruction - for parameter explanation see
        `kimimaro <https://github.com/seung-lab/kimimaro>`_"""
        dust_threshold = 600  # cubic physical units
        scale = 1
        const = 0  # physical units
        pdrf_exponent = 4
        pdrf_scale = 100000
        soma_detection_threshold = 1100  # physical units
        soma_acceptance_threshold = 3500  # physical units
        soma_invalidation_scale = 1.0
        soma_invalidation_const = 300  # physical units
        max_paths = 50  # default None
