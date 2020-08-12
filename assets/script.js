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

function checkFieldNEAT()
{
    let el

    if (popSize && (el = document.getElementById("inputNeatMaxPop")).value == "")
    {
        el.focus()
        return false
    }
    else if (genCount && (el = document.getElementById("inputNeatGen")).value == "")
    {
        el.focus()
        return false
    }
    else if (genCount && (el = document.getElementById("inputNeatDelta")).value == "")
    {
        el.focus()
        return false
    }
    else if (selNumber && (el = document.getElementById("inputNeatC1")).value == "")
    {
        el.focus()
        return false
    }
    else if (crossOver && (el = document.getElementById("inputNeatC2")).value == "")
    {
        el.focus()
        return false
    }
    else if (mutation && (el = document.getElementById("inputNeatC3")).value == "")
    {
        el.focus()
        return false
    }

    return true
}

function getNetworkType()
{
    radios = document.getElementsByName("radioNetType")
    for(let i = 0; i < radios.length; i++)
        if(radios[i].checked)
        {
            return i+1
        }
    return -1
}

function isNeatSelected()
{
    if(getNetworkType() > 4)
        return true
    else
        return false
}

function getSendString(guiCode)
{
    return `${guiCode}`
}

function addElementValueToSendString(str, field)
{
    return str + GUI_CODE_SEPARATOR + document.getElementById(field).value
}

function addValueToSendString(str, value)
{
    return str + GUI_CODE_SEPARATOR + value
}

function sendTrainData()
{
    if(isNeatSelected())
    {
        //if(checkFieldNEAT())
        //{
            str = getSendString(GUI_CODE_TRAIN)
            str = addValueToSendString(str, 
                        String(getNetworkType()))
            str = addElementValueToSendString(str, 'inputNeatGen')
            str = addElementValueToSendString(str, 'inputNeatMaxPop')
            str = addElementValueToSendString(str, 'inputNeatDelta')
            str = addElementValueToSendString(str, 'inputNeatC1')
            str = addElementValueToSendString(str, 'inputNeatC2')
            str = addElementValueToSendString(str, 'inputNeatC3')
            str = addElementValueToSendString(str, 'inputNeatSurvival')
            str = addElementValueToSendString(str, 'inputNeatReproduction')
            str = addElementValueToSendString(str, 'inputNeatBias')
            str = addElementValueToSendString(str, 'inputNeatWeight')
            str = addElementValueToSendString(str, 'inputNeatToggle')
            str = addElementValueToSendString(str, 'inputNeatNode')
            str = addElementValueToSendString(str, 'inputNeatConnection')
            
            sendMessageToJulia(str)
        //}
    }
    else
        if(checkFields(true, true, true, true, true, true))
        {
            str = getSendString(GUI_CODE_TRAIN)
            str = addValueToSendString(str, 
                        String(getNetworkType()))
            str = addElementValueToSendString(str, 'inputGenCount')
            str = addElementValueToSendString(str, 'inputPopSize')
            str = addElementValueToSendString(str, 'inputSelectionNumber')
            str = addElementValueToSendString(str, 'inputCrossoverDivisor')
            str = addElementValueToSendString(str, 'inputMutationRate')
            
            sendMessageToJulia(str)
        }
}

function sendTrainDataWithDataset()
{
    if(isNeatSelected())
    {
        str = getSendString(GUI_CODE_TRAIN_EXISTING)

        str += addElementValueToSendString(str, 'inputNeatGen')
        /*str += addElementValueToSendString(str, 'inputNeatDelta')
        str += addElementValueToSendString(str, 'inputNeatC1')
        str += addElementValueToSendString(str, 'inputNeatC2')
        str += addElementValueToSendString(str, 'inputNeatC3')
        str += addElementValueToSendString(str, 'inputNeatSurvival')
        str += addElementValueToSendString(str, 'inputNeatReproduction')
        str += addElementValueToSendString(str, 'inputNeatBias')
        str += addElementValueToSendString(str, 'inputNeatWeight')
        str += addElementValueToSendString(str, 'inputNeatToggle')
        str += addElementValueToSendString(str, 'inputNeatNode')
        str += addElementValueToSendString(str, 'inputNeatConnection')
*/
        sendMessageToJulia(str)
    }
    else
    {
    //    if(checkFields(false, true, true, true, true, false))
    //    {
            str = getSendString(GUI_CODE_TRAIN_EXISTING)

            str += addElementValueToSendString(str, 'inputGenCount')

            sendMessageToJulia(str)
     //   }
    }
}

/*//to fix
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
*/

function changeNet()
{
    if(isNeatSelected())
    {
        document.getElementById('neatForm').style.display = 'block'
        document.getElementById('defaultForm').style.display = 'none'
    }
    else
    {
        document.getElementById('neatForm').style.display = 'none'
        document.getElementById('defaultForm').style.display = 'block'
    }   
}