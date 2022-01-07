use crate::{
    util::{Load, group_barrier},
    byte::u8x4,
    short::u16x2,
    autobind,
};
use spirv_std::glam::UVec3;

#[repr(C)]
pub struct AccuracyPushConsts {
    n: u32,
}

fn accuracy<T>(
    group_id: UVec3,
    local_id: UVec3,
    x: &[T],
    t: &[T],
    y: &mut [u32],
    y_group: &mut [u32; 256],
    push_consts: &AccuracyPushConsts,
) where T: Copy, [T]: Load<u32> {
    let group_id = group_id.x as usize;
    let local_id = local_id.x as usize;
    let global_id = group_id * 256 + local_id;
    let n = push_consts.n as usize;
    y_group[local_id] = if global_id < n {
        if x.load(global_id) == t.load(global_id) {
            1
        } else {
            0
        }
    } else {
        0
    };
    group_barrier();
    if local_id == 0 {
        let mut acc = 0;
        for i in 0 .. 256 {
            acc += y_group[i];
        }
        y[group_id] = acc;
    }
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn accuracy_u8(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)]
    x: &[u8x4],
    #[spirv(storage_buffer)]
    t: &[u8x4],
    #[spirv(storage_buffer)]
    y: &mut [u32],
    #[spirv(workgroup)]
    y_group: &mut [u32; 256],
    #[spirv(push_constant)]
    push_consts: &AccuracyPushConsts,
) {
    accuracy(group_id, local_id, x, t, y, y_group, push_consts);
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn accuracy_u16(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)]
    x: &[u16x2],
    #[spirv(storage_buffer)]
    t: &[u16x2],
    #[spirv(storage_buffer)]
    y: &mut [u32],
    #[spirv(workgroup)]
    y_group: &mut [u32; 256],
    #[spirv(push_constant)]
    push_consts: &AccuracyPushConsts,
) {
    accuracy(group_id, local_id, x, t, y, y_group, push_consts);
}

#[autobind]
#[spirv(compute(threads(256)))]
pub fn accuracy_u32(
    #[spirv(workgroup_id)]
    group_id: UVec3,
    #[spirv(local_invocation_id)]
    local_id: UVec3,
    #[spirv(storage_buffer)]
    x: &[u32],
    #[spirv(storage_buffer)]
    t: &[u32],
    #[spirv(storage_buffer)]
    y: &mut [u32],
    #[spirv(workgroup)]
    y_group: &mut [u32; 256],
    #[spirv(push_constant)]
    push_consts: &AccuracyPushConsts,
) {
    accuracy(group_id, local_id, x, t, y, y_group, push_consts);
}
