macro mt(n, exp)
    quote
        for i in 1:$(esc(n))
            $(exp)
        end
    end
end
    
function ft()
    r = 0
    s = 1
    n = 1
    #i refeers to the indexer inside macro quote
    #r refeers to the ft function scope, which has to be passed
    #this way to the macro in reason of getting the variable from the correct scope
    @mt n print(:($r)+i)
    r
end

println(ft())