use dahl_partition::*;

use std::slice;

pub fn binder_single(partition: &[usize], psm: &SquareMatrixBorrower) -> f64 {
    let ni = partition.len();
    assert_eq!(ni, psm.n_items());
    let mut sum = 0.0;
    for j in 0..ni {
        for i in 0..j {
            let p = unsafe { *psm.get_unchecked((i, j)) };
            sum += if unsafe { *partition.get_unchecked(i) == *partition.get_unchecked(j) } {
                1.0 - p
            } else {
                p
            }
        }
    }
    sum
}

pub fn binder_multiple(
    partitions: &PartitionsHolderBorrower,
    psm: &SquareMatrixBorrower,
    results: &mut [f64],
) {
    let ni = partitions.n_items();
    assert_eq!(ni, psm.n_items());
    for k in 0..partitions.n_partitions() {
        let mut sum = 0.0;
        for j in 0..ni {
            for i in 0..j {
                let p = unsafe { *psm.get_unchecked((i, j)) };
                sum += if unsafe {
                    *partitions.get_unchecked((k, i)) == *partitions.get_unchecked((k, j))
                } {
                    1.0 - p
                } else {
                    p
                }
            }
        }
        unsafe { *results.get_unchecked_mut(k) = sum };
    }
}

pub fn lpear_single(partition: &[usize], psm: &SquareMatrixBorrower) -> f64 {
    let ni = partition.len();
    assert_eq!(ni, psm.n_items());
    let mut sum_p = 0.0;
    let mut sum_ip = 0.0;
    let mut sum_i = 0.0;
    for j in 0..ni {
        let cj = unsafe { *partition.get_unchecked(j) };
        for i in 0..j {
            let p = unsafe { *psm.get_unchecked((i, j)) };
            sum_p += p;
            if unsafe { *partition.get_unchecked(i) } == cj {
                sum_ip += p;
                sum_i += 1.0;
            }
        }
    }
    let no2 = (ni * (ni - 1) / 2) as f64;
    let correc = (sum_i * sum_p) / no2;
    1.0 - (sum_ip - correc) / (0.5 * (sum_p + sum_i) - correc)
}

pub fn lpear_multiple(
    partitions: &PartitionsHolderBorrower,
    psm: &SquareMatrixBorrower,
    results: &mut [f64],
) {
    let ni = partitions.n_items();
    assert_eq!(ni, psm.n_items());
    let no2 = (ni * (ni - 1) / 2) as f64;
    let mut sum_p = 0.0;
    for j in 0..ni {
        for i in 0..j {
            sum_p += unsafe { *psm.get_unchecked((i, j)) };
        }
    }
    for k in 0..partitions.n_partitions() {
        let mut sum_ip = 0.0;
        let mut sum_i = 0.0;
        for j in 0..ni {
            let cj = unsafe { *partitions.get_unchecked((k, j)) };
            for i in 0..j {
                if unsafe { *partitions.get_unchecked((k, i)) } == cj {
                    sum_ip += unsafe { *psm.get_unchecked((i, j)) };
                    sum_i += 1.0;
                }
            }
        }
        let correc = (sum_i * sum_p) / no2;
        unsafe {
            *results.get_unchecked_mut(k) =
                1.0 - (sum_ip - correc) / (0.5 * (sum_p + sum_i) - correc)
        };
    }
}

pub fn vilb_expected_loss_constant(psm: &SquareMatrixBorrower) -> f64 {
    let ni = psm.n_items();
    let mut s1: f64 = 0.0;
    for i in 0..ni {
        let mut s2: f64 = 0.0;
        for j in 0..ni {
            s2 += unsafe { *psm.get_unchecked((i, j)) };
        }
        s1 += s2.log2();
    }
    s1
}

pub fn vilb_single_kernel(partition: &[usize], psm: &SquareMatrixBorrower) -> f64 {
    let ni = partition.len();
    assert_eq!(ni, psm.n_items());
    let mut sum = 0.0;
    for i in 0..ni {
        let mut s1 = 0u32;
        let mut s3 = 0.0;
        for j in 0..ni {
            if unsafe { *partition.get_unchecked(i) == *partition.get_unchecked(j) } {
                s1 += 1;
                s3 += unsafe { *psm.get_unchecked((i, j)) };
            }
        }
        sum += f64::from(s1).log2() - 2.0 * s3.log2();
    }
    sum
}

pub fn vilb_single_kernel_for_partial_partition(
    partition: &Partition,
    psm: &SquareMatrixBorrower,
) -> f64 {
    let labels = partition.labels();
    let ni = partition.n_items();
    assert_eq!(ni, psm.n_items());
    let mut sum = 0.0;
    for i in 0..ni {
        if labels[i].is_none() {
            continue;
        }
        let mut s1 = 0u32;
        let mut s3 = 0.0;
        for j in 0..ni {
            if labels[j].is_none() {
                continue;
            }
            if partition.label_of(i) == partition.label_of(j) {
                s1 += 1;
                s3 += unsafe { *psm.get_unchecked((i, j)) };
            }
        }
        sum += f64::from(s1).log2() - 2.0 * s3.log2();
    }
    sum
}

pub fn vilb_single(partition: &[usize], psm: &SquareMatrixBorrower) -> f64 {
    (vilb_single_kernel(partition, psm) + vilb_expected_loss_constant(psm)) / (psm.n_items() as f64)
}

pub fn vilb_multiple(
    partitions: &PartitionsHolderBorrower,
    psm: &SquareMatrixBorrower,
    results: &mut [f64],
) {
    let ni = partitions.n_items();
    assert_eq!(ni, psm.n_items());
    let constant = vilb_expected_loss_constant(psm);
    for k in 0..partitions.n_partitions() {
        let mut sum = constant;
        for i in 0..ni {
            let mut s1 = 0u32;
            let mut s3 = 0.0;
            for j in 0..ni {
                if unsafe { *partitions.get_unchecked((k, i)) == *partitions.get_unchecked((k, j)) }
                {
                    s1 += 1;
                    s3 += unsafe { *psm.get_unchecked((i, j)) };
                }
            }
            sum += f64::from(s1).log2() - 2.0 * s3.log2();
        }
        unsafe { *results.get_unchecked_mut(k) = sum / (psm.n_items() as f64) };
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_salso__expected_loss(
    n_partitions: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_ptr: *mut f64,
) {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let partitions = PartitionsHolderBorrower::from_ptr(partition_ptr, np, ni, true);
    let psm = SquareMatrixBorrower::from_ptr(psm_ptr, ni);
    let results = slice::from_raw_parts_mut(results_ptr, np);
    match loss {
        0 => binder_multiple(&partitions, &psm, results),
        1 => lpear_multiple(&partitions, &psm, results),
        2 => vilb_multiple(&partitions, &psm, results),
        _ => panic!("Unsupported loss method: {}", loss),
    };
}

#[cfg(test)]
mod tests_loss {
    use super::*;

    #[test]
    fn test_computations() {
        let n_items = 5;
        let mut samples = PartitionsHolder::new(n_items);
        for labels in Partition::iter(n_items) {
            samples.push_partition(&Partition::from(&labels[..]));
        }
        let n_partitions = samples.n_partitions();
        let mut psm = crate::psm::psm(&samples.view(), true);
        let samples_view = &samples.view();
        let psm_view = &psm.view();
        let mut results = vec![0.0; n_partitions];
        binder_multiple(samples_view, psm_view, &mut results[..]);
        for i in 0..n_partitions {
            assert_relative_eq!(
                binder_single(&samples_view.get(i).labels_via_copying()[..], psm_view),
                results[i]
            );
        }
        lpear_multiple(samples_view, psm_view, &mut results[..]);
        for i in 0..n_partitions {
            assert_relative_eq!(
                npear_single(&samples_view.get(i).labels_via_copying()[..], psm_view),
                results[i]
            );
        }
        vilb_multiple(samples_view, psm_view, &mut results[..]);
        for i in 0..n_partitions {
            assert_ulps_eq!(
                vilb_single(&samples_view.get(i).labels_via_copying()[..], psm_view),
                results[i]
            );
        }
        for i in 1..n_partitions {
            assert_ulps_eq!(
                ((1.0 / (n_items as f64))
                    * (vilb_single_kernel(&samples_view.get(i).labels_via_copying()[..], psm_view)
                        - vilb_single_kernel(
                            &samples_view.get(i - 1).labels_via_copying()[..],
                            psm_view
                        ))) as f32,
                (results[i] - results[i - 1]) as f32,
            );
        }
    }
}
