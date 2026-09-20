#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("Usage: gptcelltype_runner.R QUERY OUTPUT TISSUE MODEL")
query_path <- args[[1]]
output_path <- args[[2]]
tissue_context <- args[[3]]
model_use <- args[[4]]

api_key <- Sys.getenv("OPENAI_API_KEY")
if (!nzchar(api_key)) stop("OPENAI_API_KEY is required for GPTCelltype annotation")

relay_base_url <- Sys.getenv("OPENAI_BASE_URL")
if (!nzchar(relay_base_url)) relay_base_url <- Sys.getenv("OPENAI_API_BASE")
if (nzchar(relay_base_url)) {
  relay_base_url <- sub("/+$", "", relay_base_url)
  if (grepl("/chat/completions$", relay_base_url)) {
    chat_completion_url <- relay_base_url
  } else if (grepl("/v1$", relay_base_url)) {
    chat_completion_url <- paste0(relay_base_url, "/chat/completions")
  } else {
    chat_completion_url <- paste0(relay_base_url, "/v1/chat/completions")
  }
  relay_create_chat_completion <- function(model, messages = NULL, temperature = 1, top_p = 1, n = 1, stream = FALSE, stop = NULL, max_tokens = NULL, presence_penalty = 0, frequency_penalty = 0, logit_bias = NULL, user = NULL, openai_api_key = Sys.getenv("OPENAI_API_KEY"), openai_organization = NULL) {
    headers <- c(Authorization = paste("Bearer", openai_api_key), `Content-Type` = "application/json")
    if (!is.null(openai_organization)) headers["OpenAI-Organization"] <- openai_organization
    response <- httr::POST(chat_completion_url, httr::add_headers(.headers = headers), body = list(model = model, messages = messages, temperature = temperature, top_p = top_p, n = n, stream = stream, stop = stop, max_tokens = max_tokens, presence_penalty = presence_penalty, frequency_penalty = frequency_penalty, logit_bias = logit_bias, user = user), encode = "json")
    response_text <- httr::content(response, as = "text", encoding = "UTF-8")
    parsed <- jsonlite::fromJSON(response_text, flatten = TRUE)
    if (httr::http_error(response)) stop(paste0("Relay request failed [", httr::status_code(response), "]: ", response_text), call. = FALSE)
    parsed
  }
  ns <- asNamespace("openai")
  unlockBinding("create_chat_completion", ns)
  assign("create_chat_completion", relay_create_chat_completion, envir = ns)
  lockBinding("create_chat_completion", ns)
}

query <- read.delim(query_path, check.names = FALSE, stringsAsFactors = FALSE)
if (!all(c("source_row_id", "marker") %in% names(query)) || nrow(query) < 1 || nrow(query) > 30 || anyDuplicated(query$source_row_id)) stop("Query must contain unique source_row_id and marker rows (1..30)")
markers <- strsplit(query$marker, split = "\\s*,\\s*")
names(markers) <- query$source_row_id
prediction <- GPTCelltype::gptcelltype(markers, tissuename = tissue_context, model = model_use)
if (length(prediction) != nrow(query) || !identical(names(prediction), query$source_row_id)) stop("GPTCelltype result does not align with input")
write.table(data.frame(source_row_id = query$source_row_id, prediction = unname(prediction), stringsAsFactors = FALSE), output_path, sep = "\t", row.names = FALSE, quote = TRUE)
