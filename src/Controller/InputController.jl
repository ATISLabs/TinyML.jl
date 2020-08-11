function getArgumentArray(input::String)
    msg = split(input, GUI_CODE_SEPARATOR)
    return [String(x) for x in msg]
end

getCommand(input::Array{String,1}) = input[1]
getInputArguments(input::Array{String,1}) = input[2:end]

function getNetType(code::Int)
    if code == 1
        return FloatMLP
    elseif code == 2
        return BitMLP
    elseif code == 3
        return FloatCNN
    elseif code == 4
        return BitCNN
    elseif code == 5
        return FloatNEAT
    elseif code == 6
        return BitNEAT
    end
end

@inline getNetType(arr::Array{String,1}) = getNetType(parse(Int, arr[1]))

@inline isNeat(code::Int) = code > 4

function getTrainingInput(input::Array{String,1})
    if isNeat(parse(Int, args[1]))
        return (nt, parse(Int, args[2]),
                parse(Float64, args[3]),
                parse(Float64, args[4]),
                parse(Float64, args[5]),
                parse(Float64, args[6]),
                parse(Float64, args[7]),
                parse(Float64, args[8]),
                parse(Float64, args[9]),
                parse(Float64, args[10]),
                parse(Float64, args[11]),
                parse(Float64, args[12]),
                parse(Float64, args[13]),
                parse(Float64, args[14]))
    else
        return (nt, parse(Int, args[2]),
                parse(Int, args[3]),
                parse(Int, args[4]),
                parse(Float64, args[5]))
    end
end

getGenCount(input::Tuple) = input[1]