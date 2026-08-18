use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rayon::prelude::*;

static GLOBAL_START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn log_incumbent(length: usize, elapsed_sec: f64, energy: isize, node_count: usize, seq_str: &str) {
    let filename = format!("objective_time_series_N{}.csv", length);
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(filename) {
        let _ = writeln!(file, "{:.6},{},{},{}", elapsed_sec, energy, node_count, seq_str);
    }
}

fn log_summary(length: usize, time_taken: f64, nodes_visited: usize, best_energy: isize, seq_str: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("normal_brnbo_parallel.txt")?;
    
    writeln!(
        file, 
        "N: {}, Energy: {}, Sequence: {}, Time: {:.4}s, Nodes: {}",
        length, best_energy, seq_str, time_taken, nodes_visited
    )?;
    
    Ok(())
}

fn modular_min_abs(a: isize, b: isize) -> isize {
    let mut result = a % b;
    if result > b / 2 {
        result -= b;
    } else if result < -b / 2 {
        result += b;
    }
    result.abs()
}

#[derive(Clone, Debug)]
struct BitSequence {
    sequence: Vec<bool>,
    fixed_energys: Vec<isize>,
    u_min: Vec<usize>,
    u_max: Vec<usize>,
    m: usize,
    n: usize,
    branched_pp: bool,
    branched_pm: bool,
    branched_mp: bool,
    branched_mm: bool,
}

impl BitSequence {
    fn new(length: usize) -> Self {
        BitSequence {
            sequence: Vec::new(),
            fixed_energys: Vec::new(),
            u_min: Vec::new(),
            u_max: Vec::new(),
            m: 0,
            n: length,
            branched_pp: false,
            branched_pm: false,
            branched_mp: false,
            branched_mm: false,
        }
    }

    fn to_pm_string(&self) -> String {
        self.sequence
            .iter()
            .map(|&b| if b { '+' } else { '-' })
            .collect()
    }

    fn initialize_fixed_energys(&mut self) {
        self.fixed_energys = vec![0; self.n - 1];
    }

    fn multi_update_fixed_energys(&mut self) {
        let m = self.m;
        let n = self.n;

        for k in 0..(n-1) {
            for i in 0..m {
                if i + (k+1) < m {
                    if self.sequence[i] ^ self.sequence[i + (k+1)] {
                        self.fixed_energys[k] -= 1;
                    } else {
                        self.fixed_energys[k] += 1;
                    }
                }
                if i + (k+1) >= n-m && i+(k+1) < n {
                    if self.sequence[i] ^ self.sequence[i + (k+1) - (n-2*m)] {
                        self.fixed_energys[k] -= 1;
                    } else {
                        self.fixed_energys[k] += 1;
                    }
                }
            }
            for i in 0..m {
                if n >= n - m + 1 + i + (k+1) {
                    if self.sequence[2*m - 1 - i] ^ self.sequence[2*m - 1 - i - (k+1)] {
                        self.fixed_energys[k] -= 1;
                    } else {
                        self.fixed_energys[k] += 1;
                    }
                }
            }
        }       
    }

    fn multi_update_u_min_max(&mut self) {
        self.u_max = vec![0; self.n - 1];
        self.u_min = vec![0; self.n - 1];
        
        let m = self.m;
        let n = self.n;

        for k in 0..n-m-1 {
            let base_amount = if (k+1) > n/2 { n -(k+1) } else { k+1 }; 
            let disappeared = if m > (n-(k+1))/2 { 2*m - (n-(k+1)) } else {0};
            let amount = base_amount - disappeared;

            for i in 0..amount {
                let first_chain = if m <= k+1 {0} else {m-(k+1)};
                let chain = if i + first_chain + (k+1) >= n-m { i + first_chain + disappeared } else { i + first_chain };
                let chain_length = if m <= k+1 {((n - (k+1) -1 -chain ) / (k+1)) + 1} else {( (n - m -1 -chain)/(k+1) ) + 1}; 

                if chain >= m || chain + chain_length*(k+1) < n - m {
                    self.u_max[k] += chain_length;
                    self.u_min[k] += chain_length;
                } else if chain_length % 2 == 0 {
                    if self.sequence[first_chain +i] ^ self.sequence[first_chain + i + chain_length*(k+1) - (n-2*m)] {
                        self.u_max[k] += chain_length -2;
                        self.u_min[k] += chain_length -2;
                    } else {
                        self.u_max[k] += chain_length;
                        self.u_min[k] += chain_length;
                    }
                } else if self.sequence[first_chain +i] ^ self.sequence[first_chain + i + chain_length*(k+1) - (n-2*m)] {
                    self.u_max[k] += chain_length -2;
                    self.u_min[k] += chain_length;
                } else {
                    self.u_max[k] += chain_length;
                    self.u_min[k] += chain_length -2;
                }
            }
        }
    }

    fn set_bits(&mut self, left: bool, right: bool) {
        let mid = self.m;
        self.sequence.insert(mid, left);
        self.sequence.insert(mid + 1, right);
        self.m += 1;
    }

    fn update_fixed_energys(&mut self) {
        let m = self.m;
        let n = self.n;

        for k in 0..(n - m) {
            if m + k >= n - m {
                if self.sequence[m - 1] ^ self.sequence[3 * m + k - n] {
                    self.fixed_energys[k] -= 1;
                } else {
                    self.fixed_energys[k] += 1;
                }
            }

            if m + k > n - m {
                if self.sequence[m] ^ self.sequence[n - m - (k + 1)] {
                    self.fixed_energys[k] -= 1;
                } else {
                    self.fixed_energys[k] += 1;
                }
            }

            if m >= k + 2 {
                if self.sequence[m - 1] ^ self.sequence[m - (k + 2)] {
                    self.fixed_energys[k] -= 1;
                } else {
                    self.fixed_energys[k] += 1;
                }
                if self.sequence[m] ^ self.sequence[m + (k + 1)] {
                    self.fixed_energys[k] -= 1;
                } else {
                    self.fixed_energys[k] += 1;
                }
            }
        }
    }

    fn update_u_min_max(&mut self) {
        let m = self.m;
        let n = self.n;

        self.u_max[n - m - 1] = 0;
        self.u_min[n - m - 1] = 0;

        for k in 0..(n - m - 1) {
            if m  <= k + 1 {
                let chain_length = ((n - (k+1) -m ) / (k+1)) + 1;

                if m-1 + chain_length*(k+1) == n-m {
                    if chain_length == 1 {
                        self.u_max[k] -= 1;
                        self.u_min[k] -= 1;
                        continue;
                    } else if self.sequence[m-1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                        if chain_length % 2 == 0 {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2;
                            continue; 
                        } else {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 0;
                            continue;
                        }
                    } else if chain_length % 2 == 0 {
                        self.u_max[k] -= 0;
                        self.u_min[k] -= 0; 
                        continue;
                    } else {
                        self.u_max[k] -= 0;
                        self.u_min[k] -= 2;
                        continue;
                    }
                }
            
                if m-1 + chain_length*(k+1) > n - m {
                    if chain_length == 1 {
                        self.u_max[k] -= 2;
                        self.u_min[k] -= 2;
                        continue;
                    } else if chain_length % 2 == 0 {
                        if self.sequence[m-1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2; 
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 0;
                        }
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2; 
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 0;
                        }
                        continue;
                    } else {
                        if self.sequence[m-1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 0; 
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 2;                  
                        }
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 0; 
                        } else {
                            self.u_max[k] -= 0;
                            self.u_min[k] -= 2;                 
                        }
                        continue;
                    }
                }
            } else {
                let chain_length = ( (n - 2*m)/(k+1) ) + 1;

                if m - 1 + chain_length*(k+1) == n-m {
                    if chain_length == 1 {
                        if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m) +(k+1)] {
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3;
                            continue;
                        } else {
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1;
                            continue;
                        }
                    } else if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m) +(k+1)] {
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2;
                            continue;
                        } else {
                            if chain_length % 2 == 0 {
                                self.u_max[k] -= 0;
                                self.u_min[k] -= 0;
                                continue;
                            } else {
                                self.u_max[k] -= 0;
                                self.u_min[k] -= 4;
                                continue;
                            }
                        }
                    } else {
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                            if chain_length % 2 == 0 {
                                self.u_max[k] -= 4;
                                self.u_min[k] -= 4;
                                continue;
                            } else {
                                self.u_max[k] -= 4;
                                self.u_min[k] -= 0;
                                continue;
                            }
                        } else {
                            self.u_max[k] -= 2;
                            self.u_min[k] -= 2;
                            continue;
                        }
                    }
                } else if chain_length == 1 {
                    if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                        self.u_max[k] -= 0;
                        self.u_min[k] -= 0;
                    } else {
                        self.u_max[k] -= 2;
                        self.u_min[k] -= 2;
                    }
                    if self.sequence[m + (k+1)] ^ self.sequence[n - m - chain_length*(k+1)] {
                        self.u_max[k] -= 0;
                        self.u_min[k] -= 0;
                    } else {
                        self.u_max[k] -= 2;
                        self.u_min[k] -= 2;
                    }
                    continue;
                }

                if chain_length % 2 == 0 {
                    if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3; 
                        } else {
                            self.u_max[k] += 1;
                            self.u_min[k] -= 1; 
                        }
                    } else {
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1; 
                        } else {
                            self.u_max[k] -= 1;
                            self.u_min[k] += 1;
                        }
                    }

                    if self.sequence[m + (k+1)] ^ self.sequence[n - m - chain_length*(k+1)] {
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] {
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3; 
                        } else {
                            self.u_max[k] += 1;
                            self.u_min[k] -= 1; 
                        }
                    } else {
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] {
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1; 
                        } else {
                            self.u_max[k] -= 1;
                            self.u_min[k] += 1;
                        }
                    }
                    continue;
                } else {
                    if self.sequence[m - 1 - (k+1)] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                            self.u_max[k] -= 1;
                            self.u_min[k] += 1; 
                        } else {
                            self.u_max[k] += 1;
                            self.u_min[k] -= 1;
                        }
                    } else {
                        if self.sequence[m - 1] ^ self.sequence[m - 1 + chain_length*(k+1) - (n - 2*m)] {
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1; 
                        } else {
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3;
                        }
                    }

                    if self.sequence[m + (k+1)] ^ self.sequence[n - m - chain_length*(k+1)] {
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] {
                            self.u_max[k] -= 1;
                            self.u_min[k] += 1; 
                        } else {
                            self.u_max[k] += 1;
                            self.u_min[k] -= 1;
                        }
                    } else {
                        if self.sequence[m] ^ self.sequence[n - m - chain_length*(k+1)] {
                            self.u_max[k] -= 3;
                            self.u_min[k] -= 1; 
                        } else {
                            self.u_max[k] -= 1;
                            self.u_min[k] -= 3;
                        }
                    }
                    continue;
                }
            }  
        }
    }

    fn set_last_bit(&mut self, bit: bool) {
        let mid = self.sequence.len() / 2;
        self.sequence.insert(mid, bit);
        self.m += 1;
        let n = self.n;
        let m = self.m;

        for k in 0..(n - m) {
            if self.sequence[m - 1] ^ self.sequence[m - 2 - k] {
                self.fixed_energys[k] -= 1;
            } else {
                self.fixed_energys[k] += 1;
            }
        
            if self.sequence[m - 1] ^ self.sequence[m + k] {
                self.fixed_energys[k] -= 1;
            } else {
                self.fixed_energys[k] += 1;
            }
        }   
    }
}

fn branch(sequence: &mut BitSequence) -> BitSequence {
    let mut seq = sequence.clone();

    if !sequence.branched_mm {
        seq.set_bits(false, false);
        sequence.branched_mm = true;
    } else if !sequence.branched_mp {
        seq.set_bits(false, true);
        sequence.branched_mp = true;
    } else if !sequence.branched_pm {
        seq.set_bits(true, false);
        sequence.branched_pm = true;
    } else {
        seq.set_bits(true, true);
        sequence.branched_pp = true;
    }

    seq.branched_mm = false;
    seq.branched_mp = false;
    seq.branched_pm = false;
    seq.branched_pp = false;

    seq.update_fixed_energys();
    seq.update_u_min_max();

    seq
}

fn calculate_e_min(individual: &mut BitSequence) -> isize {
    let mut result = vec![0; individual.n - 1];

    for (k, item) in result.iter_mut().enumerate().take(individual.n - 1) {
        if individual.fixed_energys[k] <= -(individual.u_max[k] as isize) {
            *item = individual.fixed_energys[k] + individual.u_max[k] as isize;
        } else if individual.fixed_energys[k] >= individual.u_min[k] as isize {
            *item = individual.fixed_energys[k] - individual.u_min[k] as isize;
        } else {
            let g_k = if individual.m < (k+1) { 2 } else { 4 };
            *item = modular_min_abs(individual.fixed_energys[k] - individual.u_min[k] as isize, g_k as isize);
        }
    }
    result.iter().map(|&x| x * x).sum()
}

fn final_energy(individual: &BitSequence) -> isize {
    individual.fixed_energys.iter().map(|&x| x * x).sum()
}

fn reverse(individual: &mut BitSequence) {
    individual.sequence.reverse();
}

fn complement(individual: &mut BitSequence) {
    for bit in &mut individual.sequence {
        *bit = !*bit;
    }
}

fn alternate_complement(individual: &mut BitSequence) {
    let mid = individual.sequence.len() / 2;
    for i in 0..mid {
        if i % 2 == 0 {
            individual.sequence[i] = !individual.sequence[i];
        }
    }
    for i in 0..mid {
        if i % 2 != individual.n % 2 {
            individual.sequence[2*mid -1 -i] = !individual.sequence[2*mid -1 -i];
        }
    }
}

/// Helper function to atomically update global best energy and record sequence
fn check_and_update_incumbent(
    new_energy: isize,
    seq: &BitSequence,
    global_best: &AtomicIsize,
    best_seq_mutex: &Mutex<Option<String>>,
    global_nodes: &AtomicUsize,
    length: usize,
) {
    let mut current_best = global_best.load(Ordering::Relaxed);
    while new_energy < current_best {
        match global_best.compare_exchange_weak(
            current_best,
            new_energy,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let elapsed = GLOBAL_START.get().unwrap().elapsed().as_secs_f64();
                let seq_str = seq.to_pm_string();
                let nodes_so_far = global_nodes.load(Ordering::Relaxed);

                {
                    let mut lock = best_seq_mutex.lock().unwrap();
                    *lock = Some(seq_str.clone());
                }

                println!(
                    "[Incumbent Update] N={} | New Best Energy={} | Time={:.4}s | Sequence={}",
                    length, new_energy, elapsed, seq_str
                );
                log_incumbent(length, elapsed, new_energy, nodes_so_far, &seq_str);
                break;
            }
            Err(actual) => {
                current_best = actual;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let length: usize = args.get(1)
        .expect("Provide N")
        .parse()
        .expect("N as usize");
    let bound: isize = args.get(2)
        .expect("Provide bound")
        .parse()
        .expect("bound as isize");

    let global_start = GLOBAL_START.get_or_init(Instant::now);
    let global_best_energy = Arc::new(AtomicIsize::new(bound));
    let best_sequence_str = Arc::new(Mutex::new(None::<String>));

    // Create time series CSV header
    let ts_filename = format!("objective_time_series_N{}.csv", length);
    if let Ok(mut f) = File::create(&ts_filename) {
        let _ = writeln!(f, "time_seconds,energy,nodes_visited,sequence");
    }

    let two_m = 10;
    let mut roots = Vec::new();
    let time_roots = Instant::now();

    for i in 0..(1 << two_m) {
        let mut sequence = Vec::new();
        for j in 0..two_m {
            sequence.push((i & (1 << j)) != 0);
        }

        let mut individual = BitSequence::new(length);
        individual.n = length;
        individual.m = two_m/2;
        individual.sequence = sequence;

        if individual.n % 2 == 1 {
            let mut r = individual.clone();
            reverse(&mut r);
            let mut c = individual.clone();
            complement(&mut c);
            let mut a = individual.clone();
            alternate_complement(&mut a);
            let mut rc = r.clone();
            complement(&mut rc);
            let mut ra = r.clone();
            alternate_complement(&mut ra);
            let mut ca = c.clone();
            alternate_complement(&mut ca);
            let mut rca = rc.clone();
            alternate_complement(&mut rca);

            let mut all_sequences = [
                individual.sequence.clone(),
                r.sequence.clone(),
                c.sequence.clone(),
                a.sequence.clone(),
                rc.sequence.clone(),
                ra.sequence.clone(),
                ca.sequence.clone(),
                rca.sequence.clone(),
            ];
            all_sequences.sort();
            if all_sequences.first().unwrap() == &individual.sequence {
                individual.initialize_fixed_energys();
                individual.multi_update_fixed_energys();
                individual.multi_update_u_min_max();
                roots.push(individual);
            }
        } else {
            let mut r = individual.clone();
            reverse(&mut r);
            let mut c = individual.clone();
            complement(&mut c);
            let mut a = individual.clone();
            alternate_complement(&mut a);
            let mut rc = r.clone();
            complement(&mut rc);
            let mut ra = r.clone();
            alternate_complement(&mut ra);
            let mut ar = a.clone();
            reverse(&mut ar);
            let mut ac = a.clone();
            complement(&mut ac);

            let mut all_sequences = [
                individual.sequence.clone(),
                r.sequence.clone(),
                c.sequence.clone(),
                a.sequence.clone(),
                rc.sequence.clone(),
                ra.sequence.clone(),
                ar.sequence.clone(),
                ac.sequence.clone(),
            ];
            all_sequences.sort();
            if all_sequences.first().unwrap() == &individual.sequence {
                individual.initialize_fixed_energys();
                individual.multi_update_fixed_energys();
                individual.multi_update_u_min_max();
                roots.push(individual);
            }
        }
    }

    let duration_roots = time_roots.elapsed();
    println!("{} Roots created in {} seconds", roots.len(), duration_roots.as_secs());

    let total_nodes = Arc::new(AtomicUsize::new(0));

    roots.par_iter_mut().for_each(|root| {
        let mut individuals = vec![root.clone()];

        while let Some(mut individual) = individuals.pop() {
            total_nodes.fetch_add(1, Ordering::Relaxed);

            let current_best = global_best_energy.load(Ordering::Relaxed);
            let e_min = calculate_e_min(&mut individual);

            if e_min >= current_best {
                continue;
            } else if individual.m == individual.n / 2 {
                if individual.n % 2 != 0 {
                    let mut child_f = individual.clone();
                    child_f.set_last_bit(false);
                    total_nodes.fetch_add(1, Ordering::Relaxed);
                    let energy_f = final_energy(&child_f);

                    check_and_update_incumbent(
                        energy_f,
                        &child_f,
                        &global_best_energy,
                        &best_sequence_str,
                        &total_nodes,
                        length,
                    );

                    let mut child_t = individual.clone();
                    child_t.set_last_bit(true);
                    total_nodes.fetch_add(1, Ordering::Relaxed);
                    let energy_t = final_energy(&child_t);

                    check_and_update_incumbent(
                        energy_t,
                        &child_t,
                        &global_best_energy,
                        &best_sequence_str,
                        &total_nodes,
                        length,
                    );
                } else {
                    let energy = calculate_e_min(&mut individual);
                    check_and_update_incumbent(
                        energy,
                        &individual,
                        &global_best_energy,
                        &best_sequence_str,
                        &total_nodes,
                        length,
                    );
                }
            } else {
                let child = branch(&mut individual);
                individuals.push(child);
            }

            if !individual.branched_pp && individual.m < individual.n / 2 {
                if individuals.is_empty() {
                    individuals.push(individual);
                } else {
                    individuals.insert(individuals.len() - 1, individual);
                }
            }
        }
    });

    let total_duration = global_start.elapsed().as_secs_f64();
    let final_nodes = total_nodes.load(Ordering::Relaxed);
    let final_best_energy = global_best_energy.load(Ordering::Relaxed);
    let final_seq = best_sequence_str.lock().unwrap().clone().unwrap_or_else(|| "NONE".to_string());

    println!("==================================================");
    println!("Exploration finished in {:.4}s across {} nodes", total_duration, final_nodes);
    println!("Best Energy Found: {}", final_best_energy);
    println!("Best Sequence:    {}", final_seq);
    println!("==================================================");

    let _ = log_summary(length, total_duration, final_nodes, final_best_energy, &final_seq);
}