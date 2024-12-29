moddef::moddef!(
    flat(pub) mod {
        cconv,
        conv,
        deconv,
        detrend for cfg(feature = "ndarray")
    }
);