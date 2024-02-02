
pub const fn is_prime(n: usize) -> bool
{
    let n_sqrt = 1 << ((n.ilog2() + 1) / 2);
    let mut m = 2;

    while m < n_sqrt
    {
        if n % m == 0
        {
            return false
        }
        m += 1
    }

    true
}

pub const fn closest_prime(x: usize) -> Option<usize>
{
    if x == 0
    {
        return None;
    }
    let mut n = x;
    let mut m = x - 1;
    while m != 0 || n != usize::MAX
    {
        if is_prime(n)
        {
            return Some(n)
        }
        if is_prime(m)
        {
            return Some(m)
        }
        n = n.saturating_add(1);
        m = m.saturating_sub(1);
    }
    None
}