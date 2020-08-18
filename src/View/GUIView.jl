jsAlert(w::GUIData, msg::String) = run(getWindow(w), string("alert('", msg, "')"))

function drawGame(d::GUIData, game::Game)
    w = getWindow(d)
    mat = getDrawingMatrix(game)

    str = "matrix = ["
    limitLine = size(mat)[1] - 1
    limitChar = size(mat)[2] - 1
    for i in 1:limitLine
        str = string(str, "[")
        for j in 1:limitChar
            str = string(str, "'", string(mat[i, j]), "',")
        end
        str = string(str, "'", string(mat[i, size(mat)[2]]), "'],")
    end
    str = string(str, "[")
    for j in 1:limitChar
        str = string(str, "'", string(mat[size(mat)[1], j]), "',")
    end
    str = string(str, "'", string(mat[size(mat)[1], size(mat)[2]]), "']]")

    run(w, str)
end

@inline function printStats(d::GUIData, currGen::Int, genCnt::Int, fit::Float64, pop::Int, net::NetworkType)
    n = string(net)
    str = "printStats('$currGen', '$genCnt', '$fit', '$pop', '$n')"
    run(getWindow(d), str)
end

@inline getInput(d::GUIData) = take!(msgchannel(getWindow(d)))