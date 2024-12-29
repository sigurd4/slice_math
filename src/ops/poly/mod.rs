pub use super::poly_signal::*;

moddef::moddef!(
    flat(pub) mod {
        derivate_polynomial,
        integrate_polynomial,
        poly_eval,
        poly_fit for cfg(feature = "ndarray"),
        poly_roots for cfg(feature = "ndarray")
    }
);