##############################################
# Shiny app (kdml-only distances) with:
# - Multiple clustering methods:
#   * Hierarchical (average/complete/single)
#   * PAM (k-medoids) on kdml distances
#   * KMeans on MDS of kdml distances
#   * Spectral clustering from kdml distances (RBF kernel)
# - Two spec modes:
#   * PRESETS (old fixed menus)
#   * CUSTOM (by counts per distribution — you pick how many of each)
# - Add optional UNINFORMATIVE columns (continuous & discrete uniform)
# - Confusion matrix (+ ARI if mclust present)
# - Save dataset CSV + HARD membership CSV + TRUE POSTERIOR CSV
##############################################

install.packages("kdml")
# ---------- Required packages ----------
need <- c("shiny","ggplot2","plotly","viridis","gridExtra","cluster","R6","mclust")
for (p in need) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, quiet = TRUE)
  suppressPackageStartupMessages(library(p, character.only = TRUE))
}
if (!requireNamespace("kdml", quietly = TRUE)) {
  stop("This app requires kdml (no fallback).\nInstall, e.g.: devtools::install_github('owner/repo') and library(kdml).")
}

# ---------- Helpers ----------
`%||%` <- function(a,b) if (!is.null(a)) a else b
enc <- function(x) if (is.factor(x) || is.ordered(x)) as.numeric(x) else x
as_num_vec <- function(txt, k = NULL, positive = FALSE) {
  v <- as.numeric(strsplit(gsub("\\s+", "", txt), ",", fixed = FALSE)[[1]])
  if (any(is.na(v))) stop("Non-numeric value found.")
  if (!is.null(k) && length(v) != k) stop("Expected length ", k, " but got ", length(v), ".")
  if (positive && any(v <= 0)) stop("Values must be positive.")
  v
}
parse_prob_matrix <- function(txt, k, n_cat) {
  rows <- strsplit(gsub("\\r", "", txt), "\n", fixed = FALSE)[[1]]
  if (length(rows) < k) stop("Provide ", k, " rows of probabilities (one per cluster).")
  lapply(seq_len(k), function(i) {
    v <- as.numeric(strsplit(gsub("\\s+", "", rows[i]), ",", fixed = FALSE)[[1]])
    if (length(v) != n_cat) stop("Row ", i, ": expected ", n_cat, " probabilities.")
    if (any(is.na(v))) stop("Row ", i, ": non-numeric probability.")
    if (any(v < 0)) stop("Row ", i, ": probabilities must be >= 0.")
    v / sum(v)
  })
}
fmt_vec <- function(v) { vf <- formatC(v, format = "g", digits = 6); paste(vf, collapse = "_") }
sanitize_token <- function(x) { x <- gsub("[^A-Za-z0-9_.-]+", "_", x); x <- gsub("_+", "_", x); x <- gsub("^_|_$", "", x); x }

make_membership <- function(cl, k) {
  n <- length(cl)
  M <- matrix(0L, nrow = n, ncol = k)
  M[cbind(seq_len(n), cl)] <- 1L
  colnames(M) <- paste0("cluster_", seq_len(k))
  M
}

# ---- TRUE posterior (row-normalized) given exact gen specs & mix ----
.compute_true_posteriors <- function(df_full, specs_used, mixing) {
  df <- df_full[, seq_along(specs_used), drop = FALSE]  # exclude noise cols
  n <- nrow(df); k <- length(mixing)
  stopifnot(k >= 2, n > 0, length(specs_used) == ncol(df))
  loglik <- matrix(rep(log(mixing), each = n), nrow = n, ncol = k)
  
  for (j in seq_along(specs_used)) {
    spec <- specs_used[[j]]
    tp   <- spec$type
    dist <- spec$distribution
    xcol <- df[[j]]
    
    if (tp == "continuous") {
      x <- as.numeric(xcol)
      if (dist == "gaussian") {
        for (kk in seq_len(k)) loglik[,kk] <- loglik[,kk] + dnorm(x, spec$params$means[kk], spec$params$stds[kk], log=TRUE)
      } else if (dist == "gamma") {
        for (kk in seq_len(k)) loglik[,kk] <- loglik[,kk] + dgamma(x, spec$params$shapes[kk], scale=spec$params$scales[kk], log=TRUE)
      } else if (dist == "beta") {
        for (kk in seq_len(k)) loglik[,kk] <- loglik[,kk] + dbeta(x, spec$params$alphas[kk], spec$params$betas[kk], log=TRUE)
      } else if (dist == "exponential") {
        for (kk in seq_len(k)) loglik[,kk] <- loglik[,kk] + dexp(x, rate=1/spec$params$scales[kk], log=TRUE)
      } else if (dist == "lognormal") {
        for (kk in seq_len(k)) loglik[,kk] <- loglik[,kk] + dlnorm(x, meanlog=spec$params$means[kk], sdlog=spec$params$sigmas[kk], log=TRUE)
      } else if (dist == "chi2") {
        for (kk in seq_len(k)) loglik[,kk] <- loglik[,kk] + dchisq(x, df=spec$params$dfs[kk], log=TRUE)
      } else stop("Unknown continuous distribution: ", dist)
      
    } else if (tp == "ordinal") {
      x <- if (is.ordered(xcol) || is.factor(xcol)) as.numeric(as.character(xcol)) else as.numeric(xcol)
      if (dist == "poisson") {
        for (kk in seq_len(k)) loglik[,kk] <- loglik[,kk] + dpois(x, lambda=spec$params$lambdas[kk], log=TRUE)
      } else if (dist == "negative_binomial") {
        for (kk in seq_len(k)) loglik[,kk] <- loglik[,kk] + dnbinom(x, size=spec$params$r_values[kk], prob=spec$params$p_values[kk], log=TRUE)
      } else if (dist == "discrete_uniform") {
        for (kk in seq_len(k)) {
          lo <- spec$params$lows[kk]; hi <- spec$params$highs[kk]
          inside <- (x >= lo & x <= hi)
          val <- if (hi >= lo) log(1/(hi-lo+1)) else -Inf
          loglik[,kk] <- loglik[,kk] + ifelse(inside, val, -Inf)
        }
      } else stop("Unknown ordinal distribution: ", dist)
      
    } else if (tp == "nominal") {
      cat_idx <- if (is.factor(xcol)) as.integer(xcol) else as.integer(xcol)
      for (kk in seq_len(k)) {
        probs <- pmax(spec$params$prob_matrices[[kk]], .Machine$double.eps)
        loglik[,kk] <- loglik[,kk] + log(probs[cat_idx])
      }
    } else stop("Unknown feature type: ", tp)
  }
  
  row_max <- apply(loglik, 1, max)
  Z <- exp(loglik - row_max)
  Z <- Z / pmax(rowSums(Z), .Machine$double.eps)
  colnames(Z) <- paste0("cluster_", seq_len(k))
  Z
}

# ---------- R6: generator (kdml distance + multiple clusterers) ----------
FuzzyMixtureGenerator <- R6::R6Class(
  "FuzzyMixtureGenerator",
  public = list(
    seed = NULL,
    initialize = function(seed = 42) { self$seed <- seed; set.seed(seed) },
    
    generate_dataset = function(feature_specs, n_components, sample_size,
                                mixing_proportions = NULL, apply_overlap = FALSE, overlap_level = 0.5) {
      
      private$validate_inputs(feature_specs, n_components, sample_size, mixing_proportions)
      if (is.null(mixing_proportions)) mixing_proportions <- rep(1/n_components, n_components)
      stopifnot(abs(sum(mixing_proportions) - 1) < 1e-6)
      
      if (isTRUE(apply_overlap)) {
        cat(sprintf("[GEN] Applying overlap (level=%.2f) to input params...\n", overlap_level)); flush.console()
      }
      cat(sprintf("[GEN] Simulating data (k=%d, N=%d, dims=%d)...\n",
                  n_components, sample_size, length(feature_specs))); flush.console()
      
      specs <- if (isTRUE(apply_overlap)) private$adjust_overlap(feature_specs, n_components, overlap_level) else feature_specs
      
      # True assignments
      true_k <- sample(seq_len(n_components), size = sample_size,
                       prob = mixing_proportions, replace = TRUE)
      
      # Simulate data
      p <- length(specs)
      X <- matrix(NA_real_, nrow = sample_size, ncol = p)
      for (j in seq_len(p)) for (i in seq_len(sample_size))
        X[i, j] <- private$generate_feature_value(specs[[j]], true_k[i])
      
      df <- as.data.frame(X)
      colnames(df) <- vapply(seq_along(specs), function(i) specs[[i]]$name %||% paste0("Feature_", i), character(1))
      for (i in seq_along(specs)) {
        if (specs[[i]]$type == "nominal") df[[i]] <- factor(df[[i]])
        if (specs[[i]]$type == "ordinal") df[[i]] <- ordered(df[[i]])
      }
      
      list(
        data = df,
        true_clusters = true_k,
        specs_used = specs,                 # exact specs used (for posterior)
        mixing = mixing_proportions,        # true mixing weights
        k = n_components
      )
    },
    
    cluster_df = function(df, k, method, linkage = "average",
                          kmeans_mds_dim = 3, kmeans_nstart = 10,
                          spectral_sigma = NA_real_, spectral_nstart = 10) {
      private$cluster_with_kdml(df, k = k, method = method, linkage = linkage,
                                kmeans_mds_dim = kmeans_mds_dim, kmeans_nstart = kmeans_nstart,
                                spectral_sigma = spectral_sigma, spectral_nstart = spectral_nstart)
    }
  ),
  private = list(
    validate_inputs = function(feature_specs, n_components, sample_size, mixing_proportions) {
      stopifnot(is.list(feature_specs))
      if (length(feature_specs) < 1) stop("You must include at least one informative feature.")
      stopifnot(n_components >= 2, sample_size >= 1)
      if (!is.null(mixing_proportions)) stopifnot(length(mixing_proportions) == n_components)
    },
    
    # ---- Main clustering switch (always kdml for distance) ----
    cluster_with_kdml = function(df, k, method, linkage,
                                 kmeans_mds_dim, kmeans_nstart,
                                 spectral_sigma, spectral_nstart) {
      
      cat(sprintf("[CLUST] Computing kdml distance matrix (n=%d, p=%d)...\n", nrow(df), ncol(df))); flush.console()
      dres <- kdml::dkss(df = df)
      if (is.null(dres$distances) || !is.matrix(dres$distances) ||
          anyNA(dres$distances) || nrow(dres$distances) != ncol(dres$distances)) {
        stop("kdml::dkss did not return a valid square distance matrix.")
      }
      D <- dres$distances
      dist_obj <- stats::as.dist(D)
      
      method <- tolower(method)
      hclust_model <- NULL
      clusters <- NULL
      
      if (grepl("^hierarchical", method)) {
        meth <- sub("^hierarchical\\s*\\(([^)]+)\\).*", "\\1", method)
        if (!meth %in% c("average","complete","single"))
          stop("Unsupported hierarchical linkage: ", meth)
        cat(sprintf("[CLUST] hclust (method=%s) + cutree(k=%d)...\n", meth, k)); flush.console()
        hclust_model <- stats::hclust(dist_obj, method = meth)
        clusters <- stats::cutree(hclust_model, k = k)
        
      } else if (method == "pam (k-medoids)") {
        cat(sprintf("[CLUST] PAM on kdml distances (k=%d)...\n", k)); flush.console()
        pam_fit <- cluster::pam(D, k = k, diss = TRUE)
        clusters <- pam_fit$clustering
        hclust_model <- NULL
        
      } else if (method == "kmeans (mds of kdml)") {
        md <- max(1, min(kmeans_mds_dim, nrow(df) - 1))
        cat(sprintf("[CLUST] cmdscale (dim=%d) on kdml + kmeans(nstart=%d, k=%d)...\n", md, kmeans_nstart, k)); flush.console()
        Xmds <- cmdscale(dist_obj, k = md)
        km <- stats::kmeans(Xmds, centers = k, nstart = kmeans_nstart, iter.max = 100)
        clusters <- km$cluster
        hclust_model <- NULL
        
      } else if (method == "spectral (rbf from kdml)") {
        dvec <- D[upper.tri(D)]
        s <- if (is.na(spectral_sigma) || spectral_sigma <= 0) median(dvec[dvec > 0]) else spectral_sigma
        cat(sprintf("[CLUST] Spectral: building RBF kernel (sigma=%.4f) ...\n", s)); flush.console()
        W <- exp(-(D^2)/(2*s^2)); diag(W) <- 0
        deg <- rowSums(W)
        inv_sqrt_deg <- 1/sqrt(pmax(deg, .Machine$double.eps))
        Lsym <- (inv_sqrt_deg * W) * rep(inv_sqrt_deg, each = nrow(W))
        cat("[CLUST] Eigen decomposition...\n"); flush.console()
        ev <- eigen(Lsym, symmetric = TRUE)
        U <- ev$vectors[, seq_len(k), drop = FALSE]
        rn <- sqrt(rowSums(U^2)); U <- U/pmax(rn, .Machine$double.eps)
        cat(sprintf("[CLUST] kmeans on eigen-embedding (nstart=%d, k=%d)...\n", spectral_nstart, k)); flush.console()
        km <- stats::kmeans(U, centers = k, nstart = spectral_nstart, iter.max = 100)
        clusters <- km$cluster
        hclust_model <- NULL
        
      } else stop("Unknown clustering method: ", method)
      
      sil <- cluster::silhouette(clusters, dist_obj)
      sil_avg <- mean(sil[,3])
      cat(sprintf("[CLUST] Done. Avg silhouette = %.3f\n", sil_avg)); flush.console()
      
      list(clusters = clusters,
           hclust_model = hclust_model,
           distance_matrix = D,
           silhouette_avg = sil_avg,
           bandwidths = dres$bandwidths %||% rep(NA_real_, ncol(df)))
    },
    
    # ---- Overlap adjustment helpers ----
    adjust_overlap = function(feature_specs, n_components, overlap_level) {
      out <- vector("list", length(feature_specs))
      for (i in seq_along(feature_specs)) {
        spec <- feature_specs[[i]]
        adj <- spec
        if (spec$type == "continuous" && spec$distribution == "gaussian") {
          adj$params <- private$adjust_gaussian_overlap(spec$params, n_components, overlap_level)
        } else if (spec$type == "ordinal" &&
                   spec$distribution %in% c("poisson","negative_binomial","discrete_uniform")) {
          adj$params <- private$adjust_ordinal_overlap(spec$params, n_components, overlap_level)
        } else if (spec$type == "nominal") {
          adj$params <- private$adjust_nominal_overlap(spec$params, n_components, overlap_level)
        }
        out[[i]] <- adj
      }
      out
    },
    adjust_gaussian_overlap = function(params, n_components, overlap_level) {
      means <- params$means; stds <- params$stds
      mu0 <- mean(means); sep <- 1.0 - overlap_level + 0.1
      list(means = mu0 + (means - mu0) * sep,
           stds  = stds * (0.5 + 1.5 * overlap_level))
    },
    adjust_ordinal_overlap = function(params, n_components, overlap_level) {
      out <- params
      if (!is.null(params$lambdas)) {
        lam <- params$lambdas; c0 <- mean(lam); sep <- 1.0 - overlap_level + 0.1
        out$lambdas <- pmax(c0 + (lam - c0) * sep, 0.1)
      }
      if (!is.null(params$r_values)) out$r_values <- pmax(params$r_values, 0.5)
      if (!is.null(params$p_values)) out$p_values <- pmin(pmax(params$p_values, 0.05), 0.95)
      if (!is.null(params$lows) && !is.null(params$highs)) {
        out$lows  <- pmax(0, floor(params$lows  * (0.9 - 0.3 * overlap_level)))
        out$highs <- ceiling(params$highs * (1.1 + 0.3 * overlap_level))
      }
      out
    },
    adjust_nominal_overlap = function(params, n_components, overlap_level) {
      mats <- params$prob_matrices
      ncat <- length(mats[[1]]); unif <- rep(1/ncat, ncat)
      adj <- lapply(mats, function(p) { q <- (1 - overlap_level) * p + overlap_level * unif; q / sum(q) })
      list(prob_matrices = adj)
    },
    
    # ---- Sampling helpers ----
    generate_feature_value = function(spec, cluster) {
      tp <- spec$type; dist_name <- spec$distribution; p <- spec$params
      if (tp == "continuous") {
        return(switch(dist_name,
                      gaussian    = rnorm(1, mean = p$means[cluster], sd = p$stds[cluster]),
                      gamma       = rgamma(1, shape = p$shapes[cluster], scale = p$scales[cluster]),
                      beta        = rbeta(1, shape1 = p$alphas[cluster], shape2 = p$betas[cluster]),
                      exponential = rexp(1, rate = 1 / p$scales[cluster]),
                      lognormal   = rlnorm(1, meanlog = p$means[cluster], sdlog = p$sigmas[cluster]),
                      chi2        = rchisq(1, df = p$dfs[cluster]),
                      stop("Unknown continuous distribution: ", dist_name)
        ))
      }
      if (tp == "ordinal") {
        return(switch(dist_name,
                      poisson           = rpois(1, lambda = p$lambdas[cluster]),
                      negative_binomial = rnbinom(1, size = p$r_values[cluster], prob = p$p_values[cluster]),
                      discrete_uniform  = sample(p$lows[cluster]:p$highs[cluster], 1),
                      stop("Unknown ordinal distribution: ", dist_name)
        ))
      }
      if (tp == "nominal") {
        if (dist_name != "categorical") stop("Unknown nominal distribution: ", dist_name)
        return(sample.int(length(p$prob_matrices[[cluster]]), 1, prob = p$prob_matrices[[cluster]]))
      }
      stop("Unknown type: ", tp)
    }
  )
)

# ---------- PRESET dataset skeletons (for convenience) ----------
.make_rotating_probs <- function(k, n_cat = 3, main_p = 0.7) {
  lapply(seq_len(k), function(j) {
    dom <- ((j - 1) %% n_cat) + 1
    p <- rep((1 - main_p)/(n_cat - 1), n_cat); p[dom] <- main_p; p
  })
}
# 2D
spec_2d_non_gauss <- function(k) list(
  list(type="continuous", distribution="chi2",        params=list(dfs=round(seq(2,10,length.out=k))), name="X_chi2"),
  list(type="continuous", distribution="exponential", params=list(scales=seq(1.0,3.0,length.out=k)),  name="X_exp")
)
spec_2d_mixed_cont <- function(k) list(
  list(type="continuous", distribution="gaussian", params=list(means=seq(-2,2,length.out=k), stds=rep(1,k)), name="X_gauss"),
  list(type="continuous", distribution="gamma",    params=list(shapes=seq(2,6,length.out=k), scales=seq(1.5,0.9,length.out=k)), name="X_gamma")
)
spec_2d_gaussian2 <- function(k) list(
  list(type="continuous", distribution="gaussian", params=list(means=seq(-2.5,2.5,length.out=k), stds=rep(1,k)),   name="X_gauss1"),
  list(type="continuous", distribution="gaussian", params=list(means=seq(-1.5,3.0,length.out=k), stds=rep(1.1,k)), name="X_gauss2")
)
spec_2d_gauss_ord <- function(k) list(
  list(type="continuous", distribution="gaussian", params=list(means=seq(-1.5,2.5,length.out=k), stds=rep(1,k)), name="X_gauss"),
  list(type="ordinal",    distribution="poisson",  params=list(lambdas=seq(2,10,length.out=k)),                 name="Y_pois")
)
spec_2d_gauss_nom <- function(k) list(
  list(type="continuous", distribution="gaussian",   params=list(means=seq(-2,2,length.out=k), stds=rep(1,k)), name="X_gauss"),
  list(type="nominal",    distribution="categorical",params=list(prob_matrices=.make_rotating_probs(k, n_cat=3, main_p=0.8)), name="Z_cat")
)
# 3D
spec_3d_cno <- function(k) list(
  list(type="continuous", distribution="gaussian", params=list(means=seq(-2,2,length.out=k), stds=rep(1,k)), name="X_gauss"),
  list(type="nominal",    distribution="categorical", params=list(prob_matrices=.make_rotating_probs(k, n_cat=4, main_p=0.75)), name="Z_cat"),
  list(type="ordinal",    distribution="poisson",  params=list(lambdas=seq(2,10,length.out=k)), name="Y_count")
)
spec_3d_cn <- function(k) list(
  list(type="continuous", distribution="gamma",    params=list(shapes=seq(2,6,length.out=k), scales=seq(1.0,1.8,length.out=k)), name="X_gamma"),
  list(type="continuous", distribution="gaussian", params=list(means=seq(-1.5,2.5,length.out=k), stds=rep(0.9,k)), name="X_gauss"),
  list(type="nominal",    distribution="categorical", params=list(prob_matrices=.make_rotating_probs(k, n_cat=3, main_p=0.8)), name="Z_cat")
)
spec_3d_co <- function(k) list(
  list(type="continuous", distribution="beta",     params=list(alphas=seq(1.5,4.5,length.out=k), betas=rev(seq(1.5,4.5,length.out=k))), name="X_beta"),
  list(type="continuous", distribution="lognormal",params=list(means=seq(-0.2,0.6,length.out=k), sigmas=rep(0.4,k)), name="X_logn"),
  list(type="ordinal",    distribution="negative_binomial", params=list(r_values=seq(3,8,length.out=k), p_values=seq(0.65,0.35,length.out=k)), name="Y_nb")
)
spec_3d_all_cont <- function(k) list(
  list(type="continuous", distribution="gaussian", params=list(means=seq(-2,2,length.out=k), stds=rep(1,k)), name="X_gauss"),
  list(type="continuous", distribution="lognormal",params=list(means=seq(-0.3,0.7,length.out=k), sigmas=rep(0.35,k)), name="X_logn"),
  list(type="continuous", distribution="chi2",     params=list(dfs=round(seq(2,10,length.out=k))), name="X_chi2")
)
# 4D
spec_4d_c_c_n_o <- function(k) list(
  list(type="continuous", distribution="gaussian", params=list(means=seq(-2.5,2.5,length.out=k), stds=rep(1.1,k)), name="X_gauss"),
  list(type="continuous", distribution="gamma",    params=list(shapes=seq(2,5,length.out=k), scales=seq(1.2,1.8,length.out=k)), name="X_gamma"),
  list(type="nominal",    distribution="categorical", params=list(prob_matrices=.make_rotating_probs(k, n_cat=5, main_p=0.7)), name="Z_cat"),
  list(type="ordinal",    distribution="poisson",  params=list(lambdas=seq(3,12,length.out=k)), name="Y_count")
)
spec_4d_ccc_n <- function(k) list(
  list(type="continuous", distribution="gaussian", params=list(means=seq(-3,3,length.out=k), stds=rep(1,k)), name="X_gauss"),
  list(type="continuous", distribution="beta",     params=list(alphas=seq(1.2,4.2,length.out=k), betas=seq(4.2,1.2,length.out=k)), name="X_beta"),
  list(type="continuous", distribution="lognormal",params=list(means=seq(-0.4,0.8,length.out=k), sigmas=rep(0.3,k)), name="X_logn"),
  list(type="nominal",    distribution="categorical", params=list(prob_matrices=.make_rotating_probs(k, n_cat=4, main_p=0.75)), name="Z_cat")
)
spec_4d_ccc_o <- function(k) list(
  list(type="continuous", distribution="gaussian", params=list(means=seq(-2,2,length.out=k), stds=rep(0.9,k)), name="X_gauss"),
  list(type="continuous", distribution="gamma",    params=list(shapes=seq(2,7,length.out=k), scales=seq(0.9,1.6,length.out=k)), name="X_gamma"),
  list(type="continuous", distribution="chi2",     params=list(dfs=round(seq(2,12,length.out=k))), name="X_chi2"),
  list(type="ordinal",    distribution="discrete_uniform", params=list(lows=seq(0,3*(k-1),length.out=k), highs=seq(4,10,length.out=k)), name="Y_du")
)
spec_4d_all_cont <- function(k) list(
  list(type="continuous", distribution="gaussian", params=list(means=seq(-2.5,2.5,length.out=k), stds=rep(1,k)), name="X_gauss"),
  list(type="continuous", distribution="gamma",    params=list(shapes=seq(1.8,5.5,length.out=k), scales=seq(0.8,1.6,length.out=k)), name="X_gamma"),
  list(type="continuous", distribution="beta",     params=list(alphas=seq(1.2,3.5,length.out=k), betas=seq(3.5,1.2,length.out=k)), name="X_beta"),
  list(type="continuous", distribution="lognormal",params=list(means=seq(-0.4,0.9,length.out=k), sigmas=rep(0.35,k)), name="X_logn")
)

catalogs <- list(
  `2D` = list(
    "non-gaussian cont (chi2+exp)" = spec_2d_non_gauss,
    "mixed cont (gauss+gamma)"     = spec_2d_mixed_cont,
    "gaussian + gaussian"          = spec_2d_gaussian2,
    "gaussian + ordinal"           = spec_2d_gauss_ord,
    "gaussian + nominal"           = spec_2d_gauss_nom
  ),
  `3D` = list(
    "cont + nominal + ordinal"     = spec_3d_cno,
    "cont + nominal"               = spec_3d_cn,
    "cont + ordinal"               = spec_3d_co,
    "all-continuous"               = spec_3d_all_cont
  ),
  `4D` = list(
    "2cont + nominal + ordinal"    = spec_4d_c_c_n_o,
    "3cont + nominal"              = spec_4d_ccc_n,
    "3cont + ordinal"              = spec_4d_ccc_o,
    "all-continuous"               = spec_4d_all_cont
  )
)

# ---------- CUSTOM builder ----------
make_custom_specs <- function(k,
                              # continuous counts
                              n_gauss=0, n_gamma=0, n_beta=0, n_exp=0, n_logn=0, n_chi2=0,
                              # ordinal counts
                              n_pois=0, n_nb=0, n_du=0,
                              # nominal counts
                              n_cat=0, cat_n_levels=3, cat_main_p=0.75) {
  specs <- list(); idx <- 1
  jit <- function(i) (i-1)*0.15  # small offset to differentiate multiple columns
  
  # ---- Continuous ----
  for (i in seq_len(n_gauss)) {
    specs[[idx]] <- list(
      type="continuous", distribution="gaussian",
      params=list(
        means=seq(-2,2,length.out=k)+jit(i),
        stds =rep(1,k)
      ),
      name=paste0("X_gauss_",i)
    ); idx <- idx+1
  }
  for (i in seq_len(n_gamma)) {
    specs[[idx]] <- list(
      type="continuous", distribution="gamma",
      params=list(
        shapes=seq(2,6,length.out=k)+0.1*jit(i),
        scales=seq(1.5,0.9,length.out=k)
      ),
      name=paste0("X_gamma_",i)
    ); idx <- idx+1
  }
  for (i in seq_len(n_beta)) {
    specs[[idx]] <- list(
      type="continuous", distribution="beta",
      params=list(
        alphas=seq(1.5,3.5,length.out=k)+0.1*jit(i),
        betas =rev(seq(1.5,3.5,length.out=k))
      ),
      name=paste0("X_beta_",i)
    ); idx <- idx+1
  }
  for (i in seq_len(n_exp)) {
    specs[[idx]] <- list(
      type="continuous", distribution="exponential",
      params=list(scales=seq(0.8,1.6,length.out=k)+0.05*jit(i)),
      name=paste0("X_exp_",i)
    ); idx <- idx+1
  }
  for (i in seq_len(n_logn)) {
    specs[[idx]] <- list(
      type="continuous", distribution="lognormal",
      params=list(
        means =seq(-0.3,0.7,length.out=k)+0.05*jit(i),
        sigmas=rep(0.35,k)
      ),
      name=paste0("X_logn_",i)
    ); idx <- idx+1
  }
  for (i in seq_len(n_chi2)) {
    specs[[idx]] <- list(
      type="continuous", distribution="chi2",
      params=list(dfs=round(seq(2,10,length.out=k)+i-1)),
      name=paste0("X_chi2_",i)
    ); idx <- idx+1
  }
  
  # ---- Ordinal ----
  for (i in seq_len(n_pois)) {
    specs[[idx]] <- list(
      type="ordinal", distribution="poisson",
      params=list(lambdas=seq(2,10,length.out=k)+0.2*jit(i)),
      name=paste0("Y_pois_",i)
    ); idx <- idx+1
  }
  for (i in seq_len(n_nb)) {
    specs[[idx]] <- list(
      type="ordinal", distribution="negative_binomial",
      params=list(r_values=seq(3,8,length.out=k),
                  p_values=seq(0.65,0.35,length.out=k)),
      name=paste0("Y_nb_",i)
    ); idx <- idx+1
  }
  for (i in seq_len(n_du)) {
    lows  <- round(seq(0,  3*(k-1), length.out=k))
    highs <- round(seq(4, 10,       length.out=k)) + (i-1)
    specs[[idx]] <- list(
      type="ordinal", distribution="discrete_uniform",
      params=list(lows=lows, highs=pmax(highs, lows+1)),
      name=paste0("Y_du_",i)
    ); idx <- idx+1
  }
  
  # ---- Nominal ----
  for (i in seq_len(n_cat)) {
    specs[[idx]] <- list(
      type="nominal", distribution="categorical",
      params=list(prob_matrices=.make_rotating_probs(k, n_cat=cat_n_levels, main_p=cat_main_p)),
      name=paste0("Z_cat_",i)
    ); idx <- idx+1
  }
  
  if (length(specs) < 1) stop("Custom spec has 0 informative columns. Add at least one.")
  specs
}

# ---------- UI ----------
ui <- fluidPage(
  titlePanel("Mixed-Type Dataset Tuner (kdml) — Presets OR Custom (by counts)"),
  sidebarLayout(
    sidebarPanel(
      radioButtons("spec_mode", "Specification Mode",
                   choices = c("Presets","Custom (by counts)"),
                   selected = "Presets", inline = TRUE),
      
      conditionalPanel(
        "input.spec_mode === 'Presets'",
        selectInput("dim", "Preset family", choices = names(catalogs), selected = "2D"),
        uiOutput("dataset_ui")
      ),
      
      conditionalPanel(
        "input.spec_mode === 'Custom (by counts)'",
        h4("Informative columns — pick how many of each"),
        helpText("Each column gets its own parameter panel below (per-cluster vectors)."),
        strong("Continuous:"), br(),
        fluidRow(
          column(6, numericInput("n_gauss", "Gaussian", value = 1, min=0, step=1)),
          column(6, numericInput("n_gamma", "Gamma",    value = 0, min=0, step=1))
        ),
        fluidRow(
          column(6, numericInput("n_beta",  "Beta",      value = 0, min=0, step=1)),
          column(6, numericInput("n_exp",   "Exponential", value = 0, min=0, step=1))
        ),
        fluidRow(
          column(6, numericInput("n_logn",  "Lognormal", value = 0, min=0, step=1)),
          column(6, numericInput("n_chi2",  "Chi-square", value = 0, min=0, step=1))
        ),
        br(), strong("Ordinal:"), br(),
        fluidRow(
          column(6, numericInput("n_pois", "Poisson", value = 0, min=0, step=1)),
          column(6, numericInput("n_nb",   "Neg. Binom.", value = 0, min=0, step=1))
        ),
        fluidRow(
          column(6, numericInput("n_du",   "Discrete Uniform", value = 0, min=0, step=1))
        ),
        br(), strong("Nominal:"), br(),
        fluidRow(
          column(6, numericInput("n_cat", "Categorical", value = 0, min=0, step=1)),
          column(6, numericInput("cat_levels", "#Categories (all)", value = 3, min=2, step=1))
        ),
        sliderInput("cat_main_p", "Dominant category prob (nominal)", min=0.5, max=0.95, value=0.75, step=0.01)
      ),
      
      hr(),
      sliderInput("k", "Number of clusters (k)", min=2, max=7, value=3, step=1),
      numericInput("N", "Sample size", value=800, min=100, step=100),
      checkboxInput("apply_overlap", "Apply overlap adjustment to inputs", value = FALSE),
      conditionalPanel("input.apply_overlap == true",
                       sliderInput("overlap", "Overlap level", min=0, max=1, value=0.3, step=0.05)
      ),
      numericInput("seed", "Seed", value=42, min=1, step=1),
      textInput("mixing", "Mixing proportions (comma, len k; blank = uniform)", value = ""),
      
      hr(),
      h4("Add Uninformative (Noise) Columns"),
      numericInput("noise_cont_cols", "Continuous Uniform cols", value = 0, min = 0, step = 1),
      numericInput("noise_cont_min",  "U(a,b) — a", value = 0, step = 0.1),
      numericInput("noise_cont_max",  "U(a,b) — b", value = 1, step = 0.1),
      numericInput("noise_disc_cols", "Discrete Uniform cols", value = 0, min = 0, step = 1),
      numericInput("noise_disc_low",  "DU{L..H} — L", value = 0, step = 1),
      numericInput("noise_disc_high", "DU{L..H} — H", value = 10, step = 1),
      
      hr(),
      selectInput("cluster_method", "Clustering method",
                  choices = c("Hierarchical (average)",
                              "Hierarchical (complete)",
                              "Hierarchical (single)",
                              "PAM (k-medoids)",
                              "KMeans (MDS of kdml)",
                              "Spectral (RBF from kdml)"),
                  selected = "Hierarchical (average)"),
      conditionalPanel("input.cluster_method === 'KMeans (MDS of kdml)'",
                       numericInput("kmeans_mds_dim", "MDS dimensions", value = 3, min = 1, step = 1),
                       numericInput("kmeans_nstart", "kmeans nstart", value = 10, min = 1, step = 1)
      ),
      conditionalPanel("input.cluster_method === 'Spectral (RBF from kdml)'",
                       numericInput("spectral_sigma", "RBF sigma (blank/<=0 = auto)", value = NA),
                       numericInput("spectral_nstart", "kmeans nstart (on eigen-embedding)", value = 10, min = 1, step = 1)
      ),
      
      hr(),
      fluidRow(
        column(6, actionButton("regen", "Generate / Refresh Dataset", class = "btn-primary")),
        column(6, actionButton("do_cluster", "Run Clustering", class = "btn-warning"))
      ),
      actionButton("reset_defaults", "Reset Params to Defaults"),
      hr(),
      textInput("save_dir", "Save folder (must exist)", value = ""),
      fluidRow(
        column(6, actionButton("save_csv", "Save Dataset CSV")),
        column(6, downloadButton("download_csv", "Download Dataset CSV"))
      ),
      fluidRow(
        column(6, actionButton("save_membership", "Save Membership CSV")),
        column(6, downloadButton("download_membership", "Download Membership CSV"))
      ),
      fluidRow(
        column(6, actionButton("save_posterior", "Save TRUE Posterior CSV")),
        column(6, downloadButton("download_posterior", "Download TRUE Posterior CSV"))
      ),
      verbatimTextOutput("save_status"),
      hr(),
      verbatimTextOutput("metrics"),
      helpText("Enter comma-separated vectors per parameter (length = k).")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Parameters", uiOutput("params_ui")),
        tabPanel("Plots (True)", uiOutput("plot_ui_true")),
        tabPanel("Plots (Pred)", uiOutput("plot_ui_pred")),
        tabPanel("Dendrogram", plotOutput("dendro", height = 400)),
        tabPanel("Preview Data", tableOutput("head"))
      )
    )
  )
)

# ---------- Server ----------
server <- function(input, output, session) {
  
  # Initialize save_dir to working directory
  observe({ if (!nzchar(input$save_dir)) updateTextInput(session, "save_dir", value = getwd()) })
  
  # Preset dataset picker
  output$dataset_ui <- renderUI({
    req(input$dim)
    choices <- names(catalogs[[input$dim]])
    selectInput(
      "dataset", "Preset Dataset",
      choices = choices,
      selected = isolate(if (isTruthy(input$dataset) && input$dataset %in% choices) input$dataset else choices[1])
    )
  })
  
  # The "default spec" currently in the editor
  default_spec <- reactiveVal(NULL)
  
  # ---- Build default_spec from PRESETS ----
  observeEvent(list(input$spec_mode, input$dim, input$dataset, input$k), {
    req(input$k)
    if (input$spec_mode != "Presets") return()
    ds_list <- catalogs[[input$dim]]
    if (is.null(ds_list) || length(ds_list) == 0) return()
    cur_ds <- if (isTruthy(input$dataset) && input$dataset %in% names(ds_list)) input$dataset else names(ds_list)[1]
    spec_fun <- ds_list[[cur_ds]]
    if (!is.function(spec_fun)) return()
    default_spec(spec_fun(as.integer(input$k)))
  }, ignoreInit = FALSE)
  
  # ---- Build default_spec from CUSTOM counts ----
  observeEvent(list(input$spec_mode, input$k,
                    input$n_gauss, input$n_gamma, input$n_beta, input$n_exp, input$n_logn, input$n_chi2,
                    input$n_pois, input$n_nb, input$n_du,
                    input$n_cat, input$cat_levels, input$cat_main_p), {
                      req(input$k)
                      if (input$spec_mode != "Custom (by counts)") return()
                      k <- as.integer(input$k)
                      specs <- make_custom_specs(
                        k = k,
                        n_gauss = as.integer(input$n_gauss %||% 0),
                        n_gamma = as.integer(input$n_gamma %||% 0),
                        n_beta  = as.integer(input$n_beta  %||% 0),
                        n_exp   = as.integer(input$n_exp   %||% 0),
                        n_logn  = as.integer(input$n_logn  %||% 0),
                        n_chi2  = as.integer(input$n_chi2  %||% 0),
                        n_pois  = as.integer(input$n_pois  %||% 0),
                        n_nb    = as.integer(input$n_nb    %||% 0),
                        n_du    = as.integer(input$n_du    %||% 0),
                        n_cat   = as.integer(input$n_cat   %||% 0),
                        cat_n_levels = as.integer(input$cat_levels %||% 3),
                        cat_main_p   = as.numeric(input$cat_main_p %||% 0.75)
                      )
                      default_spec(specs)
                    }, ignoreInit = FALSE)
  
  # ---- Parameter editor UI (built from current default_spec) ----
  output$params_ui <- renderUI({
    specs <- default_spec(); req(specs)
    k <- input$k
    
    make_inputs_for_feature <- function(fi, spec) {
      type <- spec$type; dist <- spec$distribution; nm <- spec$name %||% paste0("Feature_", fi)
      ns <- NS(paste0("feat", fi))
      wellPanel(
        h4(sprintf("Feature %d: %s — %s", fi, nm, paste(type, dist, sep=" / "))),
        switch(type,
               "continuous" = switch(dist,
                                     "gaussian" = tagList(
                                       textInput(ns("means"), sprintf("Means (len %d)", k), value = paste(spec$params$means, collapse=",")),
                                       textInput(ns("sds"), sprintf("SDs OR Variances (len %d)", k), value = paste(spec$params$stds, collapse=",")),
                                       checkboxInput(ns("is_var"), "Treat above as variances (convert to SDs)", value = FALSE)
                                     ),
                                     "gamma" = tagList(
                                       textInput(ns("shapes"), sprintf("Shapes (a) (len %d)", k), value = paste(spec$params$shapes, collapse=",")),
                                       textInput(ns("scales"), sprintf("Scales (b) (len %d)", k), value = paste(spec$params$scales, collapse=","))
                                     ),
                                     "beta" = tagList(
                                       textInput(ns("alphas"), sprintf("Alphas (a) (len %d)", k), value = paste(spec$params$alphas, collapse=",")),
                                       textInput(ns("betas"),  sprintf("Betas (b) (len %d)", k),  value = paste(spec$params$betas,  collapse=","))
                                     ),
                                     "exponential" = textInput(ns("scales"), sprintf("Scales (len %d)", k), value = paste(spec$params$scales, collapse=",")),
                                     "lognormal" = tagList(
                                       textInput(ns("means"),  sprintf("Meanlog (len %d)", k), value = paste(spec$params$means, collapse=",")),
                                       textInput(ns("sigmas"), sprintf("Sdlog (len %d)", k),   value = paste(spec$params$sigmas, collapse=","))
                                     ),
                                     "chi2" = textInput(ns("dfs"), sprintf("Degrees of freedom (len %d)", k), value = paste(spec$params$dfs, collapse=","))
               ),
               "ordinal" = switch(dist,
                                  "poisson" = textInput(ns("lambdas"), sprintf("Lambdas (len %d)", k), value = paste(spec$params$lambdas, collapse=",")),
                                  "negative_binomial" = tagList(
                                    textInput(ns("r_values"), sprintf("r (len %d)", k), value = paste(spec$params$r_values, collapse=",")),
                                    textInput(ns("p_values"), sprintf("p (len %d)", k), value = paste(spec$params$p_values, collapse=","))
                                  ),
                                  "discrete_uniform" = tagList(
                                    textInput(ns("lows"),  sprintf("Lows (len %d)", k),  value = paste(spec$params$lows, collapse=",")),
                                    textInput(ns("highs"), sprintf("Highs (len %d)", k), value = paste(spec$params$highs, collapse=","))
                                  )
               ),
               "nominal" = {
                 n_cat <- length(spec$params$prob_matrices[[1]])
                 default_rows <- vapply(seq_len(k), function(i) paste(spec$params$prob_matrices[[i]], collapse=","), character(1))
                 tagList(
                   numericInput(ns("ncat"), "Number of categories", value = n_cat, min = 2, step = 1),
                   helpText("Enter ", k, " lines; each line has ", n_cat, " comma-separated probabilities (row will be normalized)."),
                   textAreaInput(ns("probs"), sprintf("Per-cluster probabilities (k = %d rows)", k),
                                 value = paste(default_rows, collapse = "\n"), rows = max(4, k))
                 )
               }
        )
      )
    }
    
    do.call(tagList, lapply(seq_along(specs), function(i) make_inputs_for_feature(i, specs[[i]])))
  })
  
  # reset to current default spec values (no-op if already)
  observeEvent(input$reset_defaults, { specs <- default_spec(); req(specs); default_spec(specs) })
  
  generator <- FuzzyMixtureGenerator$new(seed = 42)
  
  # ---- Parse UI -> feature_specs (reads all feature panels) ----
  ui_feature_specs <- reactive({
    specs <- default_spec(); req(specs)
    k <- input$k
    out <- vector("list", length(specs))
    for (i in seq_along(specs)) {
      spec <- specs[[i]]
      type <- spec$type; dist <- spec$distribution; nm <- spec$name %||% paste0("Feature_", i)
      ns <- function(id) paste0("feat", i, "-", id)
      params <- switch(type,
                       "continuous" = switch(dist,
                                             "gaussian" = {
                                               means <- as_num_vec(input[[ns("means")]], k = k)
                                               sds   <- as_num_vec(input[[ns("sds")]],   k = k, positive = TRUE)
                                               if (isTRUE(input[[ns("is_var")]])) sds <- sqrt(sds)
                                               list(means = means, stds = sds)
                                             },
                                             "gamma" = {
                                               shapes <- as_num_vec(input[[ns("shapes")]], k = k, positive = TRUE)
                                               scales <- as_num_vec(input[[ns("scales")]], k = k, positive = TRUE)
                                               list(shapes = shapes, scales = scales)
                                             },
                                             "beta" = {
                                               alphas <- as_num_vec(input[[ns("alphas")]], k = k, positive = TRUE)
                                               betas  <- as_num_vec(input[[ns("betas")]],  k = k, positive = TRUE)
                                               list(alphas = alphas, betas = betas)
                                             },
                                             "exponential" = {
                                               scales <- as_num_vec(input[[ns("scales")]], k = k, positive = TRUE)
                                               list(scales = scales)
                                             },
                                             "lognormal" = {
                                               means  <- as_num_vec(input[[ns("means")]],  k = k)
                                               sigmas <- as_num_vec(input[[ns("sigmas")]], k = k, positive = TRUE)
                                               list(means = means, sigmas = sigmas)
                                             },
                                             "chi2" = {
                                               dfs <- as_num_vec(input[[ns("dfs")]], k = k, positive = TRUE)
                                               list(dfs = round(dfs))
                                             }
                       ),
                       "ordinal" = switch(dist,
                                          "poisson" = { lambdas <- as_num_vec(input[[ns("lambdas")]], k = k, positive = TRUE); list(lambdas = lambdas) },
                                          "negative_binomial" = {
                                            r_values <- as_num_vec(input[[ns("r_values")]], k = k, positive = TRUE)
                                            p_values <- as_num_vec(input[[ns("p_values")]], k = k, positive = TRUE)
                                            if (any(p_values <= 0 | p_values >= 1)) stop("NB p must be in (0,1).")
                                            list(r_values = r_values, p_values = p_values)
                                          },
                                          "discrete_uniform" = {
                                            lows  <- as_num_vec(input[[ns("lows")]],  k = k)
                                            highs <- as_num_vec(input[[ns("highs")]], k = k)
                                            if (any(highs < lows)) stop("Discrete uniform: highs must be >= lows.")
                                            list(lows = round(lows), highs = round(highs))
                                          }
                       ),
                       "nominal" = {
                         n_cat <- input[[paste0("feat", i, "-ncat")]] %||% length(spec$params$prob_matrices[[1]])
                         if (is.null(n_cat) || n_cat < 2) stop("Number of categories must be >= 2.")
                         mats <- parse_prob_matrix(input[[paste0("feat", i, "-probs")]], k = k, n_cat = n_cat)
                         list(prob_matrices = mats)
                       }
      )
      out[[i]] <- list(type = type, distribution = dist, params = params, name = nm)
    }
    out
  })
  
  # ---- Generate dataset (and append noise) ----
  current <- eventReactive(input$regen, {
    cat(sprintf("[GEN] Starting dataset generation... (mode=%s, k=%d, N=%d)\n",
                input$spec_mode, input$k, input$N)); flush.console()
    set.seed(input$seed)
    mix <- if (nzchar(trimws(input$mixing))) {
      m <- as_num_vec(input$mixing, k = input$k, positive = TRUE); m / sum(m)
    } else rep(1/input$k, input$k)
    
    specs <- ui_feature_specs()
    out <- generator$generate_dataset(
      feature_specs = specs,
      n_components = input$k,
      sample_size = input$N,
      mixing_proportions = mix,
      apply_overlap = isTRUE(input$apply_overlap),
      overlap_level = if (isTRUE(input$apply_overlap)) input$overlap else 0
    )
    
    # ---- Append UNINFORMATIVE columns (not part of specs_used) ----
    nC <- as.integer(input$noise_cont_cols %||% 0)
    nD <- as.integer(input$noise_disc_cols %||% 0)
    if (!is.na(nC) && nC > 0) {
      a <- input$noise_cont_min %||% 0; b <- input$noise_cont_max %||% 1
      if (b < a) { tmp <- a; a <- b; b <- tmp }
      add <- replicate(nC, runif(nrow(out$data), min = a, max = b))
      add <- as.data.frame(add); colnames(add) <- paste0("NoiseC", seq_len(nC))
      out$data <- cbind(out$data, add)
    }
    if (!is.na(nD) && nD > 0) {
      L <- as.integer(input$noise_disc_low %||% 0); H <- as.integer(input$noise_disc_high %||% 10)
      if (H < L) { tmp <- L; L <- H; H <- tmp }
      addD <- replicate(nD, sample(L:H, nrow(out$data), replace = TRUE))
      addD <- as.data.frame(addD)
      for (jj in seq_len(ncol(addD))) addD[[jj]] <- ordered(addD[[jj]])
      colnames(addD) <- paste0("NoiseD", seq_len(nD))
      out$data <- cbind(out$data, addD)
    }
    cat("[GEN] Dataset ready.\n"); flush.console()
    out
  }, ignoreInit = FALSE)
  
  # ---- Clustering state ----
  cluster_state <- reactiveVal(NULL)
  
  observeEvent(input$do_cluster, {
    res <- current(); req(res)
    cm <- input$cluster_method
    linkage <- if (grepl("\\(complete\\)", cm, TRUE)) "complete" else if (grepl("\\(single\\)", cm, TRUE)) "single" else "average"
    cat(sprintf("[CLUST] Starting clustering (%s, k=%d)...\n", cm, input$k)); flush.console()
    clres <- generator$cluster_df(
      res$data, k = input$k, method = cm, linkage = linkage,
      kmeans_mds_dim = input$kmeans_mds_dim %||% 3,
      kmeans_nstart  = input$kmeans_nstart %||% 10,
      spectral_sigma = input$spectral_sigma %||% NA_real_,
      spectral_nstart= input$spectral_nstart %||% 10
    )
    cluster_state(list(
      method = cm,
      clusters = clres$clusters,
      hclust_model = clres$hclust_model,
      distance_matrix = clres$distance_matrix,
      silhouette_avg = clres$silhouette_avg,
      bandwidths = clres$bandwidths
    ))
    cat("[CLUST] Clustering complete.\n"); flush.console()
  })
  
  # ---- Metrics ----
  output$metrics <- renderPrint({
    res <- current(); req(res)
    cs <- cluster_state()
    if (is.null(cs)) { cat("No clustering run yet. Choose a method and click 'Run Clustering'."); return(invisible(NULL)) }
    ari <- if (requireNamespace("mclust", quietly = TRUE)) mclust::adjustedRandIndex(res$true_clusters, cs$clusters) else NA_real_
    true <- factor(res$true_clusters, levels = seq_len(input$k))
    pred <- factor(cs$clusters,      levels = seq_len(input$k))
    cm <- table(True = true, Pred = pred)
    cat(sprintf("Method: %s | k = %d\n", cs$method, input$k))
    if (!is.na(ari)) cat(sprintf("ARI: %.3f\n", ari))
    cat("Confusion matrix (counts):\n"); print(cm)
    rs <- rowSums(cm); rn <- sweep(cm, 1, pmax(rs, 1L), "/")
    cat("\nRow-normalized (recall by true cluster):\n"); print(round(rn, 3))
  })
  
  # ---- Preview data (head) ----
  output$head <- renderTable({
    res <- current(); req(res)
    df <- res$data
    df$TrueCluster <- res$true_clusters
    cs <- cluster_state()
    if (!is.null(cs)) df$PredCluster <- cs$clusters
    head(df, 10)
  }, rownames = TRUE)
  
  # ---------- Plot UI chooses based on INFORMATIVE features only ----------
  # A helper that returns only informative columns (excludes any appended noise)
  informative_df <- reactive({
    res <- current(); req(res)
    p <- length(res$specs_used)
    res$data[, seq_len(min(p, ncol(res$data))), drop = FALSE]
  })
  
  # ---- NEW: separate pages for True and Pred ----
  output$plot_ui_true <- renderUI({
    res <- current(); req(res)
    p <- length(res$specs_used)
    if (p == 2) {
      plotOutput("plot2d_true", height = 460)
    } else if (p == 3) {
      plotlyOutput("plot3d_true", height = 560)
    } else {
      pair_h <- max(560, 240 * p)
      plotOutput("plotPairs_true", height = pair_h)
    }
  })
  output$plot_ui_pred <- renderUI({
    res <- current(); req(res)
    p <- length(res$specs_used)
    if (p == 2) {
      plotOutput("plot2d_pred", height = 460)
    } else if (p == 3) {
      plotlyOutput("plot3d_pred", height = 560)
    } else {
      pair_h <- max(560, 240 * p)
      plotOutput("plotPairs_pred", height = pair_h)
    }
  })
  
  output$plot2d_true <- renderPlot({
    res <- current(); req(res)
    df <- informative_df(); xcol <- colnames(df)[1]; ycol <- colnames(df)[2]
    dfn <- df; dfn[[xcol]] <- enc(dfn[[xcol]]); dfn[[ycol]] <- enc(dfn[[ycol]])
    ggplot2::ggplot(transform(dfn, True=factor(res$true_clusters)),
                    ggplot2::aes_string(x=xcol, y=ycol, color="True")) +
      ggplot2::geom_point(alpha=0.7, size=1.8) +
      ggplot2::scale_color_viridis_d() +
      ggplot2::labs(title="Original (True clusters)", x=xcol, y=ycol, color=NULL) +
      ggplot2::theme_minimal()
  })
  output$plot2d_pred <- renderPlot({
    res <- current(); req(res)
    cs <- cluster_state(); df <- informative_df()
    xcol <- colnames(df)[1]; ycol <- colnames(df)[2]
    dfn <- df; dfn[[xcol]] <- enc(dfn[[xcol]]); dfn[[ycol]] <- enc(dfn[[ycol]])
    if (is.null(cs)) {
      ggplot2::ggplot() + ggplot2::annotate("text", x=0, y=0, label="Run clustering to view result", size=6) +
        ggplot2::theme_void() + ggplot2::labs(title="Predicted")
    } else {
      ggplot2::ggplot(transform(dfn, Pred=factor(cs$clusters)),
                      ggplot2::aes_string(x=xcol, y=ycol, color="Pred")) +
        ggplot2::geom_point(alpha=0.7, size=1.8, shape=17) +
        ggplot2::scale_color_viridis_d() +
        ggplot2::labs(title=paste0("Predicted — ", cs$method), x=xcol, y=ycol, color=NULL) +
        ggplot2::theme_minimal()
    }
  })
  output$plot3d_true <- plotly::renderPlotly({
    res <- current(); req(res)
    df <- informative_df(); xcol <- colnames(df)[1]; ycol <- colnames(df)[2]; zcol <- colnames(df)[3]
    dfn <- df; dfn[[xcol]] <- enc(dfn[[xcol]]); dfn[[ycol]] <- enc(dfn[[ycol]]); dfn[[zcol]] <- enc(dfn[[zcol]])
    cols <- viridis::viridis(length(unique(res$true_clusters)))
    plotly::plot_ly(x=dfn[[xcol]], y=dfn[[ycol]], z=dfn[[zcol]],
                    type="scatter3d", mode="markers", marker=list(size=3),
                    color=factor(res$true_clusters), colors=cols) |>
      plotly::layout(title="Original (True clusters)",
                     scene=list(xaxis=list(title=xcol), yaxis=list(title=ycol), zaxis=list(title=zcol)))
  })
  output$plot3d_pred <- plotly::renderPlotly({
    res <- current(); req(res)
    cs <- cluster_state()
    df <- informative_df(); xcol <- colnames(df)[1]; ycol <- colnames(df)[2]; zcol <- colnames(df)[3]
    dfn <- df; dfn[[xcol]] <- enc(dfn[[xcol]]); dfn[[ycol]] <- enc(dfn[[ycol]]); dfn[[zcol]] <- enc(dfn[[zcol]])
    cols <- viridis::viridis(length(unique(res$true_clusters)))
    if (is.null(cs)) {
      plotly::plot_ly(type="scatter3d", mode="markers") |>
        plotly::layout(title="Predicted — run clustering first")
    } else {
      plotly::plot_ly(x=dfn[[xcol]], y=dfn[[ycol]], z=dfn[[zcol]],
                      type="scatter3d", mode="markers", marker=list(size=3, symbol="diamond"),
                      color=factor(cs$clusters), colors=cols) |>
        plotly::layout(title=paste0("Predicted — ", cs$method),
                       scene=list(xaxis=list(title=xcol), yaxis=list(title=ycol), zaxis=list(title=zcol)))
    }
  })
  output$plotPairs_true <- renderPlot({
    res <- current(); req(res)
    df <- informative_df()
    use_cols <- colnames(df)
    dfn <- as.data.frame(lapply(df[, use_cols, drop = FALSE], enc))
    
    plist <- list(); idx <- 1
    for (i in seq_along(use_cols)) for (j in seq_along(use_cols)) {
      xi <- use_cols[j]; yi <- use_cols[i]
      if (i == j) {
        p <- ggplot2::ggplot(dfn, ggplot2::aes(x = .data[[xi]])) +
          ggplot2::geom_histogram(fill = "grey80", bins = 20, color = "white") +
          ggplot2::labs(x = xi, y = "Count")
      } else {
        p <- ggplot2::ggplot(dfn, ggplot2::aes(x=.data[[xi]], y=.data[[yi]], color=factor(res$true_clusters))) +
          ggplot2::geom_point(alpha=0.6, size=1.2) +
          ggplot2::scale_color_viridis_d() + ggplot2::labs(x=xi, y=yi, color=NULL)
      }
      p <- p + ggplot2::theme_minimal() + ggplot2::theme(legend.position="none")
      plist[[idx]] <- p; idx <- idx + 1
    }
    gridExtra::grid.arrange(grobs = plist, ncol = length(use_cols), top = "Original (True clusters)")
  })
  
  output$plotPairs_pred <- renderPlot({
    res <- current(); req(res)
    cs <- cluster_state()
    if (is.null(cs)) { plot.new(); title(main = "Predicted — run clustering first"); return(invisible(NULL)) }
    
    df <- informative_df()
    use_cols <- colnames(df)
    dfn <- as.data.frame(lapply(df[, use_cols, drop = FALSE], enc))
    
    plist <- list(); idx <- 1
    for (i in seq_along(use_cols)) for (j in seq_along(use_cols)) {
      xi <- use_cols[j]; yi <- use_cols[i]
      if (i == j) {
        p <- ggplot2::ggplot(dfn, ggplot2::aes(x = .data[[xi]])) +
          ggplot2::geom_histogram(fill = "grey80", bins = 20, color = "white") +
          ggplot2::labs(x = xi, y = "Count")
      } else {
        p <- ggplot2::ggplot(dfn, ggplot2::aes(x=.data[[xi]], y=.data[[yi]], color=factor(cs$clusters))) +
          ggplot2::geom_point(alpha=0.6, size=1.2) +
          ggplot2::scale_color_viridis_d() + ggplot2::labs(x=xi, y=yi, color=NULL)
      }
      p <- p + ggplot2::theme_minimal() + ggplot2::theme(legend.position="none")
      plist[[idx]] <- p; idx <- idx + 1
    }
    gridExtra::grid.arrange(grobs = plist, ncol = length(use_cols), top = paste0("Predicted — ", cs$method))
  })
  
  # Dendrogram
  output$dendro <- renderPlot({
    cs <- cluster_state()
    if (!is.null(cs) && !is.null(cs$hclust_model)) {
      plot(cs$hclust_model, labels = FALSE, hang = -1, main = paste0("Dendrogram (", cs$method, ")"))
      rect.hclust(cs$hclust_model, k = input$k, border = "darkgreen")
    }
  })
  
  # ---------- Filenames ----------
  build_filename <- reactive({
    specs <- default_spec(); req(specs)
    k <- input$k; tokens <- c(if (input$spec_mode=="Presets") input$dim else "CUSTOM")
    for (i in seq_along(specs)) {
      spec <- specs[[i]]; type <- spec$type; dist <- spec$distribution
      ns <- function(id) paste0("feat", i, "-", id)
      tok <- switch(type,
                    "continuous" = switch(dist,
                                          "gaussian" = { means <- as_num_vec(input[[ns("means")]], k = k); s <- as_num_vec(input[[ns("sds")]], k = k, positive = TRUE)
                                          if (isTRUE(input[[ns("is_var")]])) paste0("G_mean_", fmt_vec(means), "_var_", fmt_vec(s)) else paste0("G_mean_", fmt_vec(means), "_sd_", fmt_vec(s)) },
                                          "gamma" = { a <- as_num_vec(input[[ns("shapes")]], k = k, positive = TRUE); b <- as_num_vec(input[[ns("scales")]], k = k, positive = TRUE); paste0("Ga_a_", fmt_vec(a), "_b_", fmt_vec(b)) },
                                          "beta"  = { a <- as_num_vec(input[[ns("alphas")]], k = k, positive = TRUE); b <- as_num_vec(input[[ns("betas")]],  k = k, positive = TRUE); paste0("Be_a_", fmt_vec(a), "_b_", fmt_vec(b)) },
                                          "exponential" = { s <- as_num_vec(input[[ns("scales")]], k = k, positive = TRUE); paste0("Exp_s_", fmt_vec(s)) },
                                          "lognormal"   = { m <- as_num_vec(input[[ns("means")]],  k = k); sg<- as_num_vec(input[[ns("sigmas")]], k = k, positive = TRUE); paste0("Ln_m_", fmt_vec(m), "_s_", fmt_vec(sg)) },
                                          "chi2"        = { df <- as_num_vec(input[[ns("dfs")]], k = k, positive = TRUE); paste0("Chi2_df_", fmt_vec(df)) }
                    ),
                    "ordinal" = switch(dist,
                                       "poisson" = { lam <- as_num_vec(input[[ns("lambdas")]], k = k, positive = TRUE); paste0("Pois_", fmt_vec(lam)) },
                                       "negative_binomial" = { r <- as_num_vec(input[[ns("r_values")]], k = k, positive = TRUE); p <- as_num_vec(input[[ns("p_values")]], k = k, positive = TRUE); paste0("NB_r_", fmt_vec(r), "_p_", fmt_vec(p)) },
                                       "discrete_uniform"  = { lo <- as_num_vec(input[[ns("lows")]],  k = k); hi <- as_num_vec(input[[ns("highs")]], k = k); paste0("DU_", fmt_vec(lo), "_", fmt_vec(hi)) }
                    ),
                    "nominal" = { n_cat <- input[[paste0("feat", i, "-ncat")]] %||% length(spec$params$prob_matrices[[1]]); paste0("Cat_", n_cat) }
      )
      tokens <- c(tokens, tok)
    }
    if ((input$noise_cont_cols %||% 0) > 0) tokens <- c(tokens, paste0("NoiseC", as.integer(input$noise_cont_cols)))
    if ((input$noise_disc_cols %||% 0) > 0) tokens <- c(tokens, paste0("NoiseD", as.integer(input$noise_disc_cols)))
    fname <- sanitize_token(paste(tokens, collapse = "_"))
    paste0(fname, ".csv")
  })
  posterior_filename <- reactive({ paste0(sub("\\.csv$", "", build_filename()), "_true_posterior.csv") })
  membership_filename <- reactive({ paste0(sub("\\.csv$", "", build_filename()), "_membership.csv") })
  
  # ---------- Download/Save ----------
  output$download_csv <- downloadHandler(
    filename = function() build_filename(),
    content = function(file) {
      res <- current(); req(res)
      df <- res$data
      df$TrueCluster <- res$true_clusters
      cs <- cluster_state(); if (!is.null(cs)) df$PredCluster <- cs$clusters
      write.csv(df, file, row.names = FALSE)
    }
  )
  observeEvent(input$save_csv, {
    dirp <- input$save_dir; res <- current()
    if (is.null(res)) { output$save_status <- renderText("No dataset to save."); return() }
    if (!dir.exists(dirp)) { output$save_status <- renderText("Directory does not exist."); return() }
    file <- file.path(dirp, build_filename())
    df <- res$data; df$TrueCluster <- res$true_clusters
    cs <- cluster_state(); if (!is.null(cs)) df$PredCluster <- cs$clusters
    tryCatch({ write.csv(df, file, row.names = FALSE); output$save_status <- renderText(paste0("Saved dataset CSV to: ", file)) },
             error=function(e) output$save_status <- renderText(paste0("Error saving dataset CSV: ", e$message)))
  })
  
  output$download_membership <- downloadHandler(
    filename = function() membership_filename(),
    content = function(file) {
      res <- current(); cs <- cluster_state()
      if (is.null(res) || is.null(cs)) stop("Run clustering first.")
      write.csv(make_membership(cs$clusters, input$k), file, row.names = FALSE)
    }
  )
  observeEvent(input$save_membership, {
    dirp <- input$save_dir; res <- current(); cs <- cluster_state()
    if (is.null(res) || is.null(cs)) { output$save_status <- renderText("Run clustering first to save membership."); return() }
    if (!dir.exists(dirp)) { output$save_status <- renderText("Directory does not exist."); return() }
    file <- file.path(dirp, membership_filename())
    tryCatch({ write.csv(make_membership(cs$clusters, input$k), file, row.names = FALSE)
      output$save_status <- renderText(paste0("Saved membership CSV to: ", file)) },
      error=function(e) output$save_status <- renderText(paste0("Error saving membership CSV: ", e$message)))
  })
  
  true_posterior_mat <- reactive({
    res <- current(); req(res)
    cat("[POST] Computing true posterior membership...\n"); flush.console()
    .compute_true_posteriors(res$data, res$specs_used, res$mixing)
  })
  output$download_posterior <- downloadHandler(
    filename = function() posterior_filename(),
    content  = function(file) { write.csv(true_posterior_mat(), file, row.names = FALSE) }
  )
  observeEvent(input$save_posterior, {
    dirp <- input$save_dir; res <- current()
    if (is.null(res)) { output$save_status <- renderText("Generate a dataset first to save TRUE posterior."); return() }
    if (!dir.exists(dirp)) { output$save_status <- renderText("Directory does not exist."); return() }
    file <- file.path(dirp, posterior_filename())
    tryCatch({ write.csv(true_posterior_mat(), file, row.names = FALSE)
      output$save_status <- renderText(paste0("Saved TRUE posterior CSV to: ", file)) },
      error=function(e) output$save_status <- renderText(paste0("Error saving TRUE posterior CSV: ", e$message)))
  })
}

# ---------- Launch ----------
shinyApp(ui, server)
