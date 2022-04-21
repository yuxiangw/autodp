m = ExactGaussianMechanism(sigma, name = 'GM')
				compose = Composition()
				composed_mech = compose([gm], [coeff])