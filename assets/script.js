matrix = [[' ', ' ', 'X'], [' ', ' ', 'X'],['O', ' ', ' ']]

canvas = document.getElementById("canvas")
cWidth = canvas.width
cHeight = canvas.height
ctx = canvas.getContext("2d")

function drawGame()
{
    pixelWidth = cWidth / matrix[0].length
    pixelHeight = cHeight / matrix.length

    for(let i = 0; i < matrix.length; i++)
        for(let j = 0; j < matrix[0].length; j++)
        {
            if(matrix[i][j] == DRAW_MATRIX_VOID)
                ctx.fillStyle = "#000000"
            else if(matrix[i][j] == DRAW_MATRIX_SNAKE)
                ctx.fillStyle = "#008000"
            else
                ctx.fillStyle = "#ff0000"

            ctx.fillRect(pixelWidth * j, pixelHeight * i, pixelWidth, pixelHeight)            
        }
    window.requestAnimationFrame(drawGame)
}

function checkFields(popSize, genCount, selNumber, crossOver, mutation, netType)
{
    let el

    if (popSize && (el = document.getElementById("inputPopSize")).value == "")
    {
        el.focus()
        return false
    }
    else if (genCount && (el = document.getElementById("inputGenCount")).value == "")
    {
        el.focus()
        return false
    }
    else if (selNumber && (el = document.getElementById("inputSelectionNumber")).value == "")
    {
        el.focus()
        return false
    }
    else if (crossOver && (el = document.getElementById("inputCrossoverDivisor")).value == "")
    {
        el.focus()
        return false
    }
    else if (mutation && (el = document.getElementById("inputMutationRate")).value == "")
    {
        el.focus()
        return false
    }
    else
    {
        if(netType)
        {
            let radios = document.getElementsByName("radioNetType")
            for(let radio of radios)
                if(radio.checked)
                    return true
            alert("Please, choose one of Network types")
            return false
        }
    }

    return true
}

function sendTrainData()
{
    if(checkFields(true, true, true, true, true, true))
    {
        str = GUI_CODE_TRAIN + GUI_CODE_SEPARATOR +
            document.getElementById("inputPopSize").value + GUI_CODE_SEPARATOR +
            document.getElementById("inputGenCount").value + GUI_CODE_SEPARATOR +
            document.getElementById("inputSelectionNumber").value + GUI_CODE_SEPARATOR + 
            document.getElementById("inputCrossoverDivisor").value + GUI_CODE_SEPARATOR +
            document.getElementById("inputMutationRate").value + GUI_CODE_SEPARATOR

        radios = document.getElementsByName("radioNetType")
        for(let i = 0; i < radios.length; i++)
            if(radios[i].checked)
            {
                str += (i + 1)
                break;
            }

        sendMessageToJulia(str)
    }
}

function sendTrainDataWithDataset()
{
    if(checkFields(false, true, true, true, true, false))
    {
        str = GUI_CODE_TRAIN_EXISTING + GUI_CODE_SEPARATOR +
            document.getElementById("inputGenCount").value + GUI_CODE_SEPARATOR +
            document.getElementById("inputSelectionNumber").value + GUI_CODE_SEPARATOR + 
            document.getElementById("inputCrossoverDivisor").value + GUI_CODE_SEPARATOR +
            document.getElementById("inputMutationRate").value + GUI_CODE_SEPARATOR

        sendMessageToJulia(str)
    }
}

function printStats(currGen, bestFit, popSize, genCount, selNumber, crossov, net, mutation)
{
    document.getElementById('lblCurrentGen').innerHTML = currGen
    document.getElementById('lblBestFitness').innerHTML = bestFit
    document.getElementById('lblPopulationSize').innerHTML = popSize
    document.getElementById('lblGenerationCount').innerHTML = genCount
    document.getElementById('lblSelectionNumber').innerHTML = selNumber
    document.getElementById('lblCrossoverDivisorFactor').innerHTML = crossov
    document.getElementById('lblNetworkType').innerHTML = net
    document.getElementById('lblMutationRate').innerHTML = mutation
}
