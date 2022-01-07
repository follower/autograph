use super::*;
use crate::rust_shaders;

#[allow(dead_code)]
impl<S: Data<Elem = u32>, D: Dimension> TensorBase<S, D> {
    pub(crate) fn accuracy<U: Uint>(&self, target: &TensorView<U, D>) -> Result<Tensor0<u32>> {
        let device = self.device();
        let mut output = Tensor::zeros(device, ())?;
        self.accuracy_with(target, &mut output.view_mut())?;
        Ok(output)
    }
    pub(crate) fn accuracy_with<U2: Uint>(
        &self,
        target: &TensorView<U2, D>,
        output: &mut TensorViewMut0<u32>,
    ) -> Result<()> {
        let n = self.len() as u32;
        let entry = format!("accuracy::accuracy_{}", U2::scalar_name());
        let input = self.to_slice()?;
        let target = target.to_slice()?;
        let builder = rust_shaders::core()?
            .compute_pass(entry)?
            .slice(input.as_slice())?
            .slice(target.as_slice())?
            .push(n)?;
        if n <= 256 {
            let builder = builder.slice_mut(output.as_raw_slice_mut())?;
            unsafe {
                builder.submit([n, 1, 1])?;
            }
        } else {
            let tmp_len = n / 256 + if n % 256 != 0 { 1 } else { 0 };
            let mut tmp = unsafe { Tensor::<u32, _>::alloc(output.device(), tmp_len as usize)? };
            let builder = builder.slice_mut(tmp.as_raw_slice_mut())?;
            unsafe {
                builder.submit([n, 1, 1])?;
            }
            tmp.sum_with(&mut output.view_mut())?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "device_tests", feature = "learn"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn accuracy_u32() -> Result<()> {
        let x_vec = (0..1033).into_iter().collect::<Vec<u32>>();
        let t_vec = x_vec
            .iter()
            .map(|x| if x % 3 == 0 { *x } else { *x + 1 })
            .collect::<Vec<u32>>();
        let acc_true = x_vec
            .iter()
            .zip(t_vec.iter())
            .filter(|(x, t)| x == t)
            .count();
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = Tensor::from(
            Slice::from(x_vec.as_slice())
                .into_device(device.clone())
                .await?,
        );
        let t = Tensor::from(Slice::from(t_vec.as_slice()).into_device(device).await?);
        let acc = x.accuracy(&t.view())?.as_raw_slice().read().await?[0] as usize;
        assert_eq!(acc, acc_true);
        Ok(())
    }
}
