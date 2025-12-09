function split_by_lengths(vector, lengths)
    result = Vector{typeof(vector)}(undef, length(lengths))
    start_idx = 1

    for (i, len) in enumerate(lengths)
        result[i] = vector[start_idx:(start_idx + len - 1)]
        start_idx += len
    end

    return result
end
