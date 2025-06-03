
obstacle = []

end_obst = []

egg = []


upper_left_corner = []
lower_left_corner = []
upper_right_corner = []
lower_right_corner = []

small_goal = []
big_goal = []


while True:
 with open('Position2.txt', 'r') as file:
     for line in file:
        line = line.strip()
        if not line:
            continue  # spring over tomme linjer

        # Del linje i nøgle og værdi
        if ': ' not in line:
            print(f"Advarsel: Linje mangler ': ': {line}")
            continue
        # Del linjen ved kolon og fjern overskydende mellemrum
        key, value = line.strip().split(':')
        value_clean = value.replace('(', '').replace(')', '').strip()
        value = value.strip()
        if ',' not in value_clean:
            print(f"Advarsel: Ingen komma i '{value_clean}'")
            continue
        x_str, y_str = value.strip('()').split(',')
        print (x_str, y_str)
        x = int(x_str.strip('()'))
        y = int(y_str.strip())
        # Tildel værdier baseret på nøgle
        if key == 'Upper_left_corner':
            
            upper_left_corner = [int(x), int(y)]
            print ("Upper left corner:", upper_left_corner)
        elif key == 'Lower_left_corner':
            lower_left_corner = [int(x), int(y)]  
            print ("Lower left corner:", lower_left_corner)
        elif key == 'Upper_right_corner':
            upper_right_corner = [int(x), int(y)]
            print ("Upper right corner:", upper_right_corner)
        elif key == 'Lower_right_corner':
            lower_right_corner = [int(x), int(y)]
            print ("Lower right corner:", lower_right_corner)
        elif key == 'Small_goal':
            small_goal = [int(x), int(y)]
            print ("Small goal:", small_goal)
        elif key == 'Big_goal':
            big_goal = [int(x), int(y)]
            print ("Big goal:", big_goal) 
        elif key == 'Obstacle':
            obstacle = [int(x), int(y)]
        
  
 print ("Upper left corner:", upper_left_corner)
 print ("Lower left corner:", lower_left_corner)
 print ("Upper right corner:", upper_right_corner)
 print ("Lower right corner:", lower_right_corner)
 print ("Small goal:", small_goal)
 print ("Big goal:", big_goal)   

  