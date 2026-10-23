// Batched float64 Parratt with std threads. Args: <layers file> <B> <threads> <out file or -> <nq> <reps>
// Layers file: one medium per line "re im d sigma", ambient first. Prints seconds per batch.
use std::{env, fs, thread, time::Instant};

#[derive(Clone, Copy)]
struct C(f64, f64);
impl C {
    fn add(self, o: C) -> C {
        C(self.0 + o.0, self.1 + o.1)
    }
    fn sub(self, o: C) -> C {
        C(self.0 - o.0, self.1 - o.1)
    }
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn div(self, o: C) -> C {
        let n = o.0 * o.0 + o.1 * o.1;
        C(
            (self.0 * o.0 + self.1 * o.1) / n,
            (self.1 * o.0 - self.0 * o.1) / n,
        )
    }
    fn scale(self, s: f64) -> C {
        C(self.0 * s, self.1 * s)
    }
    fn exp(self) -> C {
        let e = self.0.exp();
        C(e * self.1.cos(), e * self.1.sin())
    }
    fn sqrt(self) -> C {
        let r = self.0.hypot(self.1);
        // Avoid cancellation in (r - re) when the imaginary part is small.
        if self.0 >= 0.0 {
            let re = ((r + self.0) / 2.0).sqrt();
            C(re, self.1 / (2.0 * re))
        } else {
            let im = ((r - self.0) / 2.0).sqrt().copysign(self.1);
            C(self.1 / (2.0 * im), im)
        }
    }
}

fn curve(q: &[f64], sld: &[C], d: &[f64], sig: &[f64], kz: &mut [C], out: &mut [f64]) {
    let n = sld.len();
    let pi4 = 4.0 * std::f64::consts::PI;
    for (iq, &qv) in q.iter().enumerate() {
        let h = qv / 2.0;
        for j in 0..n {
            // conj(sld - ambient) * 1e-6
            let rho = C((sld[j].0 - sld[0].0) * 1e-6, -(sld[j].1 - sld[0].1) * 1e-6);
            kz[j] = C(h * h, 0.0).sub(rho.scale(pi4)).sqrt();
        }
        let mut x = C(0.0, 0.0);
        for j in (0..n - 1).rev() {
            let (k1, k2) = (kz[j], kz[j + 1]);
            let r = k1
                .sub(k2)
                .div(k1.add(k2))
                .mul(k1.mul(k2).scale(-2.0 * sig[j + 1] * sig[j + 1]).exp());
            let dj = if j + 1 < n - 1 { d[j + 1] } else { 0.0 };
            let p = C(0.0, 2.0 * dj).mul(k2).exp();
            let xp = x.mul(p);
            x = r.add(xp).div(C(1.0, 0.0).add(r.mul(xp)));
        }
        out[iq] = x.0 * x.0 + x.1 * x.1;
    }
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let text = fs::read_to_string(&a[1]).unwrap();
    let b: usize = a[2].parse().unwrap();
    let threads: usize = a[3].parse().unwrap();
    let (mut sld, mut d, mut sig) = (vec![], vec![], vec![]);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        sld.push(C(v[0], v[1]));
        d.push(v[2]);
        sig.push(v[3]);
    }
    let nq: usize = a[5].parse().unwrap();
    let reps: usize = a[6].parse().unwrap();
    let lam = 1.5406_f64;
    let q: Vec<f64> = (0..nq)
        .map(|i| {
            let tth = 0.05 + (8.0 - 0.05) * i as f64 / (nq - 1) as f64;
            4.0 * std::f64::consts::PI / lam * (tth.to_radians() / 2.0).sin()
        })
        .collect();
    let mut out = vec![0.0; b * nq];
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..reps {
            if threads == 1 {
                let mut kz = vec![C(0.0, 0.0); sld.len()];
                for c in out.chunks_mut(nq) {
                    curve(&q, &sld, &d, &sig, &mut kz, c);
                }
                continue;
            }
            let chunk = b.div_ceil(threads) * nq;
            thread::scope(|s| {
                for part in out.chunks_mut(chunk) {
                    let (q, sld, d, sig) = (&q, &sld, &d, &sig);
                    s.spawn(move || {
                        let mut kz = vec![C(0.0, 0.0); sld.len()];
                        for c in part.chunks_mut(nq) {
                            curve(q, sld, d, sig, &mut kz, c);
                        }
                    });
                }
            });
        }
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    println!("{best}");
    if a[4] != "-" {
        let s: Vec<String> = out[..nq].iter().map(|v| format!("{v:.17e}")).collect();
        fs::write(&a[4], s.join("\n")).unwrap();
    }
}
