import numpy as np
import pytest
from vision_ros2.task_grounding import ground_box


def test_depth_localizes_only_coherent_supported_surface():
    args=([.25,.25,.75,.75],(100,100),[100,0,50,0,100,50,0,0,1],np.eye(3),np.zeros(3))
    result=ground_box(*args,depth=np.full((100,100),5.))
    assert result['position']==[0.,0.,5.]
    depth=np.full((100,100),np.nan)
    assert ground_box(*args,depth=depth)['state']=='bearing_only'
    depth[:,:50]=2.;depth[:,50:]=8.
    assert ground_box(*args,depth=depth)['depth_reason']=='mixed_depth_surfaces'
    assert ground_box(*args)['state']=='bearing_only'


def test_invalid_box_cannot_create_position():
    with pytest.raises(ValueError):
        ground_box([0,0,2,1],(100,100),np.eye(3),np.eye(3),np.zeros(3))


def test_coarse_direction_uses_camera_axes_without_object_position():
    from vision_ros2.task_grounding import ground_direction
    # Optical z -> world x, optical x -> world -y, optical y -> world -z.
    rotation=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    args=((300,600),[300,0,300,0,300,150,0,0,1],rotation,np.array([2.,3.,1.]))
    left=ground_direction('left',*args)
    right=ground_direction('right',*args)
    assert left['direction'][1]>0 and right['direction'][1]<0
    assert 'position' not in left and 'depth_m' not in left
    assert 0<left['bearing_half_angle_rad']<np.pi/2
    with pytest.raises(ValueError):ground_direction('unknown',*args)
