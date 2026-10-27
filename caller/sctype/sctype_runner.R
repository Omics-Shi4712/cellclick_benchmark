# ScType benchmark bridge. It calls the reference functions from the pinned
# Conda installation while keeping benchmark labels outside this process.
validate_text <- function(x, name, unique_required = FALSE) {
    if (!length(x) || anyNA(x) || any(!nzchar(x)) ||
        (unique_required && anyDuplicated(x))) stop(paste("Invalid", name))
}

read_expression <- function(path) {
    attrs <- lapply(rhdf5::h5readAttributes(path, "/"), as.vector)
    if (!identical(attrs$schema, "ai_cta_sctype_expression") ||
        attrs$schema_version != 1 || !identical(attrs$orientation, "genes_by_cells") ||
        !identical(attrs$normalization, "log1p(target_sum * raw_count / cell_library_size)")) {
        stop("Unsupported ScType expression schema or normalization")
    }
    read <- function(name) as.vector(rhdf5::h5read(path, name))
    shape <- read("expression/shape")
    data <- read("expression/data")
    indices <- read("expression/indices")
    pointers <- read("expression/indptr")
    genes <- as.character(read("genes"))
    cells <- as.character(read("meta/cell_id"))
    tissue <- as.character(read("meta/tissue"))
    group <- as.character(read("meta/source_row_id"))
    populations <- as.character(read("populations/source_row_id"))
    validate_text(genes, "genes", TRUE)
    validate_text(cells, "cell IDs", TRUE)
    validate_text(tissue, "tissue")
    validate_text(group, "population IDs")
    validate_text(populations, "populations", TRUE)
    if (length(unique(tissue)) != 1 || length(shape) != 2 ||
        !all(shape == c(length(genes), length(cells))) ||
        length(group) != length(cells) || !setequal(group, populations)) {
        stop("ScType expression dimensions or population mapping mismatch")
    }
    if (length(pointers) != shape[2] + 1 || pointers[1] != 0 ||
        any(diff(pointers) < 0) || tail(pointers, 1) != length(data) ||
        length(indices) != length(data) || any(indices < 0 | indices >= shape[1]) ||
        any(!is.finite(data)) || any(data < 0)) stop("Malformed CSC matrix")
    expression <- Matrix::sparseMatrix(i = indices + 1, p = pointers, x = data,
        dims = shape, dimnames = list(genes, cells), giveCsparse = TRUE)
    list(expression = expression, group = group, tissue = tissue[1])
}

scale_statistics <- function(expression) {
    means <- Matrix::rowMeans(expression)
    squares <- Matrix::rowSums(expression * expression)
    sd <- sqrt((squares / ncol(expression) - means ^ 2) /
        (ncol(expression) - 1) * ncol(expression))
    list(mean = means, sd = sd)
}

main <- function(args) {
    if (length(args) != 7) stop("Expected EXPRESSION_H5 DATABASE_XLSX DATABASE_SHA256 DATABASE_TISSUE BLOCK_SIZE OUTPUT_TSV WORK_DIR")
    suppressPackageStartupMessages(library(dplyr))
    suppressPackageStartupMessages(library(HGNChelper))
    suppressPackageStartupMessages(library(openxlsx))
    sctype_dir <- file.path(Sys.getenv("CONDA_PREFIX"), "opt", "sc-type-v1.0")
    source(file.path(sctype_dir, "R", "gene_sets_prepare.R"))
    source(file.path(sctype_dir, "R", "sctype_score_.R"))
    block_size <- as.integer(args[5])
    if (is.na(block_size) || block_size < 1) stop("Invalid block size")
    input <- read_expression(args[1])
    gs_list <- gene_sets_prepare(args[2], args[4])
    gs <- gs_list$gs_positive
    gs2 <- gs_list$gs_negative
    if (!length(gs)) stop("ScType database produced no positive marker sets")
    expression <- input$expression
    rownames(expression) <- toupper(rownames(expression))
    if (anyDuplicated(rownames(expression))) stop("ScType expression has duplicate case-insensitive gene names")
    requested <- unique(c(unlist(gs), unlist(gs2)))
    expression <- expression[intersect(requested, rownames(expression)), , drop = FALSE]
    if (!nrow(expression)) stop("ScType expression has no overlap with the database")
    statistics <- scale_statistics(expression)
    totals <- NULL
    for (start in seq.int(1, ncol(expression), by = block_size)) {
        index <- start:min(ncol(expression), start + block_size - 1)
        block <- as.matrix(expression[, index, drop = FALSE])
        block <- sweep(block, 1, statistics$mean, "-")
        block <- sweep(block, 1, statistics$sd, "/")
        scores <- sctype_score(scRNAseqData = block, scaled = TRUE, gs = gs, gs2 = gs2)
        if (!nrow(scores) || any(!is.finite(scores))) stop("ScType produced non-finite candidate scores")
        aggregated <- rowsum(t(scores), input$group[index], reorder = FALSE)
        if (is.null(totals)) {
            totals <- matrix(0, nrow = length(unique(input$group)), ncol = ncol(aggregated),
                dimnames = list(sort(unique(input$group)), colnames(aggregated)))
        } else {
            if (!identical(colnames(totals), colnames(aggregated))) stop("ScType candidate labels changed between blocks")
        }
        totals[rownames(aggregated), ] <- totals[rownames(aggregated), , drop = FALSE] + aggregated
    }
    if (is.null(totals) || any(!is.finite(totals))) stop("ScType produced no finite population scores")
    winner <- apply(totals, 1, which.max)
    result <- data.frame(source_row_id = rownames(totals), prediction = colnames(totals)[winner],
        n_cells = as.integer(table(input$group)[rownames(totals)]), stringsAsFactors = FALSE)
    dir.create(args[7], recursive = TRUE, showWarnings = FALSE)
    write.table(result, args[6], sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(totals, file.path(args[7], "summed_scores.tsv"), sep = "\t", quote = FALSE, col.names = NA)
    writeLines(capture.output(sessionInfo()), file.path(args[7], "sessionInfo.txt"))
    jsonlite::write_json(list(protocol_version = "sctype_benchmark_v1", database_tissue = args[4],
        aggregation = "sum_scores", input_normalization = "CP10K_log1p_then_gene_zscore",
        cell_block_size = block_size, R = R.version.string),
        file.path(args[7], "protocol.json"), pretty = TRUE, auto_unbox = TRUE)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
