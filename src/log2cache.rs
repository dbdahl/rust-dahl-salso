use crate::*;

#[derive(Debug, Clone)]
pub struct Log2Cache {
    log2n: Vec<f64>,
    nlog2n: Vec<f64>,
    nlog2n_difference: Vec<f64>,
}

impl Log2Cache {
    pub fn new(n: usize) -> Self {
        let mut log2n = Vec::with_capacity(n + 1);
        let mut nlog2n = Vec::with_capacity(n + 1);
        let mut nlog2n_difference = Vec::with_capacity(n + 1);
        log2n.push(0.0);
        nlog2n.push(0.0);
        for i in 1..=n {
            let i = i as f64;
            let log2i = i.log2();
            log2n.push(log2i);
            let ilog2i = i * log2i;
            let ilog2i_last = *nlog2n.last().unwrap();
            nlog2n.push(ilog2i);
            nlog2n_difference.push(ilog2i - ilog2i_last);
        }
        // Like the other tables, nlog2n_difference covers 0..=n.
        let i = (n + 1) as f64;
        nlog2n_difference.push(i * i.log2() - *nlog2n.last().unwrap());
        Self {
            log2n,
            nlog2n,
            nlog2n_difference,
        }
    }

    pub fn plog2p(&self, x: CountType, n: CountType) -> f64 {
        let p = (x as f64) / (n as f64);
        let log2p = self.log2n[x as usize] - self.log2n[n as usize];
        p * log2p
    }

    pub fn nlog2n(&self, n: CountType) -> f64 {
        self.nlog2n[n as usize]
    }

    pub fn nlog2n_difference(&self, n: CountType) -> f64 {
        self.nlog2n_difference[n as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_cover_zero_through_n() {
        let n = 5;
        let cache = Log2Cache::new(n);
        let f = |k: f64| if k == 0.0 { 0.0 } else { k * k.log2() };
        for k in 0..=n {
            let x = k as f64;
            assert_ulps_eq!(cache.nlog2n(k as CountType), f(x));
            assert_ulps_eq!(cache.nlog2n_difference(k as CountType), f(x + 1.0) - f(x));
        }
        assert_ulps_eq!(cache.plog2p(2, 4), -0.5);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn nlog2n_rejects_count_beyond_cache() {
        let _ = Log2Cache::new(4).nlog2n(60000);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn nlog2n_difference_rejects_count_beyond_cache() {
        let _ = Log2Cache::new(4).nlog2n_difference(5);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn plog2p_rejects_count_beyond_cache() {
        let _ = Log2Cache::new(4).plog2p(1, 5);
    }
}
