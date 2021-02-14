macro generateOnce(granted::Bool, exp)
    if granted
        return esc(exp)
    end
    return nothing
end