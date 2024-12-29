pub use slice_ops::ops::*;

moddef::moddef!(
    pub mod {
        poly,
        signal
    },
    mod {
        poly_signal
    },
    flat(pub) mod {
        conj_assign,
        frac_rotate,
        into_matrix for cfg(feature = "ndarray"),
        recip_assign,
        trim_zeros
    },
);