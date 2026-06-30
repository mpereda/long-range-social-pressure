# Export the CKM (Coleman, Katz & Menzel 1966) physician network data
# from the spatialprobit R package to CSV files for use in Python.

suppressMessages(library(spatialprobit))

data(CKM)
data(A1)
data(A2)
data(A3)

cat("=== CKM covariates ===\n")
str(CKM)

out_dir <- "."

write.csv(CKM, file = file.path(out_dir, "CKM_covariates.csv"), row.names = FALSE)
cat(sprintf("Wrote CKM_covariates.csv  [%d rows x %d cols]\n", nrow(CKM), ncol(CKM)))

for (nm in c("A1", "A2", "A3")) {
  m <- as.matrix(get(nm))
  cat(sprintf("\n=== %s ===\n", nm))
  cat(sprintf("dim: %d x %d, nnz: %d, is binary: %s\n",
              nrow(m), ncol(m), sum(m != 0), all(m %in% c(0, 1))))
  write.csv(m, file = file.path(out_dir, paste0(nm, ".csv")), row.names = FALSE)
  cat(sprintf("Wrote %s.csv\n", nm))
}

cat("\nDone.\n")
