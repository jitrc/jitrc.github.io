import numpy as np

# --- corpus -----------------------------------------------------------------
# 12 paraphrases per recurring mode, 10 deliberately idiosyncratic one-offs.
MODES = {
    "validation": [
        "missing required field 'date_range' in tool arguments",
        "tool call rejected: 'currency' is not a valid enum value",
        "argument 'limit' must be an integer, got string",
        "required parameter 'account_id' was not provided",
        "schema validation failed: unexpected property 'filters'",
        "invalid date format in 'start_date', expected ISO 8601",
        "tool arguments failed validation: 'threshold' out of range",
        "missing required field 'table' when calling query tool",
        "bad argument type for 'n_rows': expected int, received float",
        "tool rejected the call because 'region' is not one of the allowed values",
        "parameter 'sql' is required but was empty",
        "validation error: unknown field 'group_by' passed to the tool",
    ],
    "timeout": [
        "query exceeded the 300 second statement timeout",
        "upstream warehouse API timed out after 30 seconds",
        "tool call did not return before the deadline",
        "request cancelled: exceeded maximum execution time",
        "connection to the database timed out during handshake",
        "the export job ran past its time budget and was killed",
        "API call hung and hit the client timeout",
        "read timed out waiting for the remote endpoint",
        "operation exceeded 120s and was aborted by the supervisor",
        "the streaming response stalled until the socket timeout fired",
        "warehouse query took too long and was cancelled",
        "deadline exceeded while fetching the report",
    ],
    "auth": [
        "403 forbidden from the warehouse: insufficient permissions",
        "access token expired and refresh failed",
        "401 unauthorized when calling the upstream API",
        "service account lacks the required role on the dataset",
        "credentials rejected by the identity provider",
        "permission denied reading the restricted table",
        "OAuth token is no longer valid, re-authentication required",
        "the API key has been revoked",
        "not authorized to access this bucket",
        "authentication failed: invalid client secret",
        "insufficient scope for the requested operation",
        "access denied by the row-level security policy",
    ],
    "parse": [
        "could not parse the JSON response from the tool",
        "unexpected response schema: expected array, got object",
        "the model emitted a truncated tool call payload",
        "response contained invalid UTF-8 and failed to decode",
        "could not deserialize the CSV: inconsistent column count",
        "tool returned malformed XML",
        "failed to parse the timestamp returned by the API",
        "unexpected end of JSON input",
        "the response body did not match the declared content type",
        "could not decode base64 payload from the tool result",
        "JSON schema mismatch in the tool output",
        "the model produced a tool call with an unterminated string",
    ],
}

ONE_OFFS = [
    "the CSV had a byte order mark that shifted the first column name",
    "the model called the calculator tool with a date string as the operand",
    "a leap-year bug made February 29th parse as March 1st",
    "the spreadsheet had a merged cell that confused the range reader",
    "the tool's timezone was set to Antarctica/Troll and shifted the report by two hours",
    "the user asked in Welsh and the tool's locale lookup returned null",
    "an emoji in a product name broke the fixed-width column alignment",
    "the model invented a fifth quarter, Q5, and the aggregator accepted it",
    "a duplicate column named the same as the index silently overwrote it",
    "the report contained a number so large it lost precision as a float",
]

texts, truth = [], []
for mode, samples in MODES.items():
    for s in samples:
        texts.append(s)
        truth.append(mode)
for s in ONE_OFFS:
    texts.append(s)
    truth.append("oneoff")
truth = np.array(truth)

