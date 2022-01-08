use crate::{
    util::Load,
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
    push_consts: &AccuracyPushConsts,
) where T: Copy, [T]: Load<u32> {
    let group_id = group_id.x as usize;
    let local_id = local_id.x as usize;
    let n = push_consts.n as usize;
    if local_id == 0 {
        let mut acc = 0;
        let mut index = group_id * 256;
        let end = if (group_id + 1) * 256 < n {
            (group_id + 1) * 256
        } else {
            n
        };
        while index < end {
            if x.load(index) == t.load(index) {
                acc += 1;
            }
            index += 1;
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
    #[spirv(push_constant)]
    push_consts: &AccuracyPushConsts,
) {
    accuracy(group_id, local_id, x, t, y, push_consts);
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
    #[spirv(push_constant)]
    push_consts: &AccuracyPushConsts,
) {
    accuracy(group_id, local_id, x, t, y, push_consts);
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
    #[spirv(push_constant)]
    push_consts: &AccuracyPushConsts,
) {
    accuracy(group_id, local_id, x, t, y, push_consts);
}
