use std::ops::{Add, Mul};

pub fn lerp<T>(a: T, b: T, alpha: f32) -> T
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Mul<f32, Output = T>,
{
    a * (1.0 - alpha) + b * alpha
}

#[derive(Copy, Clone)]
pub struct ComplementaryFilter<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Mul<f32, Output = T> + Copy,
{
    /// y_(k - 1)
    current_value: T,
    alpha: f32,
}

impl<T> ComplementaryFilter<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Mul<f32, Output = T> + Copy,
{
    /// Note: To save binary space, this function does not
    ///       assert that `initial_value` is between 0 and 1.
    ///       For correct use, please assert this condition in your own code.
    pub fn new(initial_value: T, alpha: f32) -> Self {
        ComplementaryFilter {
            current_value: initial_value,
            alpha: alpha,
        }
    }

    /// Note: To save binary space, this function does not
    ///       assert that `new_alpha` is between 0 and 1.
    ///       For correct use, please assert this condition in your own code.
    pub fn set_alpha(&mut self, new_alpha: f32) {
        self.alpha = new_alpha;
    }

    /// Note: This assumes that T's multiplication by another T
    ///       is commutative and that T's multiplication by an `f32`
    ///       is also commutative.
    pub fn predict_next(&mut self, value: T) -> T {
        let y_k = (value * self.alpha) + (self.current_value * (1.0 - self.alpha));
        self.current_value = y_k;
        y_k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Quaternion;

    fn eq(a: f32, b: f32) -> bool {
        let epsilon = 0.01;
        (a - b).abs() < epsilon
    }

    #[test]
    fn works_with_f32() {
        let initial_value = 1.0;
        let alpha = 0.8;
        let x_k = 2.0;

        let mut comp_filter: ComplementaryFilter<f32> =
            ComplementaryFilter::new(initial_value, alpha);

        assert!(eq(comp_filter.predict_next(x_k), 1.8));
    }

    #[test]
    fn works_with_quaternion() {
        let initial_value = Quaternion::identity();
        let alpha = 0.8;
        let x_k: Quaternion<f32> = Quaternion::new(1.0, 2.0, 3.0, 4.0);

        let mut comp_filter: ComplementaryFilter<Quaternion<f32>> =
            ComplementaryFilter::new(initial_value, alpha);

        let y_k = comp_filter.predict_next(x_k);
        assert!(eq(y_k.i, 1.6));
        assert!(eq(y_k.j, 2.4));
        assert!(eq(y_k.k, 3.2));
        assert!(eq(y_k.w, 1.0));
    }

    #[test]
    fn lerp_works_with_f32() {
        assert!(eq(lerp(1.0, 2.0, 0.8), 1.8));
    }
}
