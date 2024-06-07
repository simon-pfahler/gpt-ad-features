import gpt as g

def parallel_transport(path, U, field):
    for (mu, distance) in path.path:
        if distance > 0:
            gauge_link = g.adj(U[mu])
            for _ in range(distance):
                field = g.cshift(gauge_link * field, mu, -1)
        else:
            gauge_link = U[mu]
            for _ in range(-distance):
                field = gauge_link * g.cshift(field, mu, 1)
    return field
