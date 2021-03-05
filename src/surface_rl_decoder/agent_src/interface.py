

def select_actions(state, agent, batch_size = 1):
        
        size = state.shape[1]-1
        ran = random.random()

        

        
        self.eps = self.eps*self.eps_decay
        if ran < eps:
            action_to_do = torch.randint(0, size*size*3+1, batch_size)
            action(action_to_do, batch_size)
        else:
            action_values = agent(state[0], state[1], state[2], batch_size)
            action_to_do = argmax(action_values, dim = 1)
            action(action_to_do)
            

        


def action(action_to_do, batch_size):
    