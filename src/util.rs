
pub const fn is_prime(n: usize) -> bool
{
    let n_sqrt = 1 << ((n.ilog2() + 1) / 2);
    let mut m = 2;

    if n <= 1
    {
        return false;
    }

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

pub const fn closest_mod0_of(mut x: usize, y: usize) -> Option<usize>
{
    if is_prime(y)
    {
        return Some(y);
    }
    if x == 0
    {
        x += 1;
    }
    let mut n = x;
    let mut m = x - 1;
    while m != 0 || n != usize::MAX
    {
        if y % n == 0
        {
            return Some(n)
        }
        if m != 0 && y % m == 0
        {
            return Some(m)
        }
        n = n.saturating_add(1);
        m = m.saturating_sub(1);
    }
    None
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