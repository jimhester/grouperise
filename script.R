pkgload::load_all()
size <- 1e7
n_groups <- 1000L
#range <- as.numeric(1:100)
range <- 1:100

g_ord <- rep(seq_len(n_groups), each = size / n_groups) - 1L
set.seed(42)
val_ord <- sample(range, size = size, replace = TRUE)

mix <- sample(seq_len(size))
val_unord <- val_ord[mix]
g_unord <- g_ord[mix]

bench::mark(
  .Call(grouped_sum1, val_unord, g_unord, n_groups),
  .Call(grouped_sum1, val_ord, g_ord, n_groups)
)
