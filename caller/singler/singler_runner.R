# SingleR TS paper protocol. No normalization or evaluation occurs here.
validate_text <- function(x, name, unique_required = FALSE) {
    if (!length(x) || anyNA(x) || any(!nzchar(x)) ||
        (unique_required && anyDuplicated(x))) stop(paste("Invalid", name))
}

read_expression <- function(path, expected_tissue) {
    attrs <- lapply(rhdf5::h5readAttributes(path, "/"), as.vector)
    if (!identical(attrs$schema, "ai_cta_singler_expression") ||
        attrs$schema_version != 2 || attrs$target_sum != 10000 ||
        !identical(attrs$orientation, "genes_by_cells") ||
        !identical(attrs$normalization, "log2(1 + target_sum * raw_count / cell_library_size)")) {
        stop("Unsupported SingleR expression schema or normalization")
    }
    listing <- rhdf5::h5ls(path)
    if (any(listing$name == "manual_celltype")) stop("Expression contains forbidden manual labels")
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
    if (length(unique(tissue)) != 1 || tissue[1] != expected_tissue) stop("Expected exactly one matching tissue")
    if (length(shape) != 2 || any(!is.finite(shape)) || any(shape <= 0) ||
        any(shape != floor(shape)) || !all(shape == c(length(genes), length(cells))) ||
        length(tissue) != length(cells) || length(group) != length(cells) ||
        !setequal(group, populations)) stop("Expression dimensions or population mapping mismatch")
    if (length(pointers) != shape[2] + 1 || any(!is.finite(pointers)) ||
        any(pointers != floor(pointers)) || pointers[1] != 0 ||
        any(diff(pointers) < 0) || tail(pointers, 1) != length(data) ||
        length(indices) != length(data) || any(!is.finite(indices)) ||
        any(indices != floor(indices)) || any(indices < 0 | indices >= shape[1]) ||
        any(!is.finite(data)) || any(data < 0)) stop("Malformed CSC matrix")
    expr <- Matrix::sparseMatrix(i = indices + 1, p = pointers, x = data,
        dims = shape, dimnames = list(genes, cells), giveCsparse = TRUE)
    if (any(Matrix::colSums(expr) <= 0)) stop("Zero-expression cells")
    list(expr = expr, group = setNames(group, cells))
}

aggregate_scores <- function(scores, group) {
    if (is.null(rownames(scores)) || anyDuplicated(rownames(scores)) ||
        !setequal(names(group), rownames(scores)) || any(!is.finite(scores))) {
        stop("Scores do not align with cells or contain non-finite values")
    }
    summed <- rowsum(scores[names(group), , drop = FALSE], group, reorder = TRUE)
    winner <- apply(summed, 1, which.max)
    list(scores = summed, predictions = data.frame(
        source_row_id = rownames(summed), prediction = colnames(summed)[winner],
        n_cells = as.integer(table(group)[rownames(summed)]), stringsAsFactors = FALSE))
}

main <- function(args) {
    if (length(args) != 7) stop("Expected EXPRESSION_H5 HPCA_RDS OUTPUT_TSV WORK_DIR WORKERS SAVE_FULLPRED TISSUE")
    suppressPackageStartupMessages(library(SingleR))
    if (as.character(packageVersion("SingleR")) != "1.4.1") stop("SingleR 1.4.1 is required")
    workers <- as.integer(args[5])
    if (is.na(workers) || workers < 1) stop("Invalid workers")
    if (!args[6] %in% c("true", "false")) stop("Invalid SAVE_FULLPRED")
    input <- read_expression(args[1], args[7])
    ref <- readRDS(args[2])
    if (!methods::is(ref, "SummarizedExperiment") ||
        !"logcounts" %in% SummarizedExperiment::assayNames(ref)) stop("HPCA requires logcounts")
    labels <- as.character(SummarizedExperiment::colData(ref)[["label.main"]])
    validate_text(labels, "HPCA label.main")
    validate_text(rownames(ref), "HPCA genes", TRUE)
    if (length(labels) != ncol(ref)) stop("HPCA labels do not align")
    if (!length(intersect(rownames(input$expr), rownames(ref)))) stop("No common genes")
    bp <- if (workers == 1) BiocParallel::SerialParam() else BiocParallel::SnowParam(workers, type = "SOCK")
    on.exit(BiocParallel::bpstop(bp), add = TRUE)
    rt <- system.time({
        pred <- SingleR(test = input$expr, ref = ref, assay.type.test = 1,
            labels = labels, BPPARAM = bp)
    })
    scores <- pred$scores
    rownames(scores) <- rownames(pred)
    result <- aggregate_scores(scores, input$group)
    dir.create(args[4], recursive = TRUE, showWarnings = FALSE)
    if (args[6] == "true") saveRDS(pred, file.path(args[4], "fullpred.rds"))
    if (args[6] == "false") unlink(file.path(args[4], "fullpred.rds"))
    saveRDS(rt, file.path(args[4], "time.rds"))
    saveRDS(result$scores, file.path(args[4], "summed_scores.rds"))
    write.table(result$predictions, args[3], sep = "\t", quote = FALSE, row.names = FALSE)
    writeLines(capture.output(sessionInfo()), file.path(args[4], "sessionInfo.txt"))
    packages <- c("SingleR", "SummarizedExperiment", "Matrix", "rhdf5", "jsonlite", "BiocParallel")
    versions <- setNames(lapply(packages, function(p) as.character(packageVersion(p))), packages)
    jsonlite::write_json(list(protocol_version = "singler_ts_paper_v1", R = R.version.string,
        packages = versions, reference_labels = "label.main", aggregation = "sum_scores",
        tie_break = "first_label", scope = "per_tissue", target_sum = 10000,
        log_base = 2, num_workers = workers, save_fullpred = args[6] == "true"),
        file.path(args[4], "protocol.json"), pretty = TRUE, auto_unbox = TRUE)
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
