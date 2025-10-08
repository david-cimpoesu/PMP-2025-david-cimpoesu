import numpy as np

# a

urna = ["red"]*3 + ["blue"]*4 + ["black"]*2;
print("Urna intiala: ", urna);

die = np.random.randint(1,7)
print("S-a dat cu zarul: ", die);

if die in [2, 3, 5]:
    urna.append("black")
    print("Am adaugat o bila neagra in urna.")
elif die == 6:
    urna.append("red")
    print("Am adaugat o bila rosie in urna.")
else:
    urna.append("blue")
    print("Am adaugat o bila albastra in urna.")

# --b--
N = 10000
count_red = 0

for _ in range(N):
    urn = ["red"] * 3 + ["blue"] * 4 + ["black"] * 2
    die = np.random.randint(1, 7)
    if die in [2, 3, 5]:
        urn.append("black")
    elif die == 6:
        urn.append("red")
    else:
        urn.append("blue")
    if np.random.choice(urn) == "red":
        count_red += 1

p_est = count_red / N
print("\nProbabilitatea ESTIMATA: ", p_est)

# --c--
# P(red) = P(red|six)*P(six) + P(red|not_six)*P(not_six)

p_red_given_six = 4/10
p_red_not_given_six = 3/10
p_not_six = 5/6
p_six =1/6

p_red = p_red_given_six * p_six + p_red_not_given_six * p_not_six
print("\nProbabilitatea TEORETICA: ", p_red)

