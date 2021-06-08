struct Connection
    in::Int
    out::Int

    W::Float32
    enabled::Bool
end

Connection(in::Int, out::Int; 
        W::Float32 = rand_weight(), 
        enabled::Bool = true) =
    Connection(
        in,
        out,
        W,
        enabled
    )

Connection(con::Connection; 
        W::Float32=W(con),
        enabled::Bool=enabled(con)) = 
    Connection(con.in,
        con.out,
        W,
        enabled
    )

#Getters and Setters
@inline in(con::Connection) = con.in
@inline out(con::Connection) = con.out
@inline W(con::Connection) = con.W
@inline enabled(con::Connection) = con.enabled