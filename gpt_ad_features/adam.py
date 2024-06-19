import gpt as g

class adam:
    def __init__(self, params, f, alpha=1e-3, beta1=0.9,
                 beta2=0.999, eps=1e-8):
        self.params = params
        self.f = f
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.reset()

    def step(self):
        self.t += 1
        self.cost = self.f(self.params)

        grad = [e.gradient for e in self.params]
        grad2 = []
        for gi in grad:
            gir = g(g.component.real(gi))
            gii = g(g.component.imag(gi))
            grad2.append(g(g.component.multiply(gir, gir) + 1j * g.component.multiply(gii, gii)))

        self.m = [g(g(self.beta1 * mi) + g((1 - self.beta1) * gi)) 
                  for mi,gi in zip(self.m, grad)]
        self.v = [g(g(self.beta2 * vi) + g((1 - self.beta2) * gi2))
                  for vi,gi2 in zip(self.v, grad2)]
        mhat = [g(mi / (1 - self.beta1**self.t)) for mi in self.m]
        vhat = [g(vi / (1 - self.beta2**self.t)) for vi in self.v]

        for p, mi, vi, epsi in zip(self.params, mhat, vhat, self.eps_converted):
            p.value -= g.component.multiply(g(self.alpha*mi), g.component.inv(g(g.component.sqrt(vi) + epsi)))

        gradnorm = sum(g.norm2(gi) for gi in grad)

        return self.params, (self.cost, self.t, gradnorm)

    def reset(self):
        # initialize iteration counter
        self.t = 0

        # initialize m and v
        self.m = [g(0*e.value) for e in self.params]
        self.v = [g(0*e.value) for e in self.params]
        
        # initialize converted eps
        self.eps_converted = [vi.new() for vi in self.v]
        for epsi in self.eps_converted:
            epsi[:] = self.eps

    def optimize(self, tol=1e-8, maxiter=10000, logging=False, reset=True):
        if reset:
            self.reset()

        costs = list()
        while self.t < maxiter:
            step_res = self.step()
            costs.append(step_res[1][0])
            
            if logging:
                print(f"{self.t}: Cost: {self.cost}, GradNorm: {step_res[1][2]}")

            if step_res[1][2] < tol:
                if logging:
                    print(f"Converged at iteration {self.t}, gradnorm = {step_res[1][2]}")
                break
        
        return self.params, (costs, self.t, step_res[1][2])
