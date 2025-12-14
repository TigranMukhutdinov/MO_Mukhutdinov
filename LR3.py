import gymnasium as gym

#Создание среды Taxi-v3
env = gym.make('Taxi-v3', render_mode='human')

#Сброс среды для получения начального состояния
observation, info = env.reset()

#Выполнение 10 случайных действий
for step in range(10):
    #Отображение среды (визуализация)
    env.render()

    #Выбор случайного действия
    action = env.action_space.sample()

    #Шаг в среде
    observation, reward, terminated, truncated, info = env.step(action)

    #done - это terminated или truncated
    done = terminated or truncated

    #Вывод информации
    print(f"Временной шаг: {step + 1}")
    print(f"Состояние (код): {observation}")
    print(f"Действие (код): {action}")
    print(f"Вознаграждение: {reward}")
    print(f"Завершено (terminated): {terminated}")
    print(f"Ограничено (truncated): {truncated}")
    print("-" * 40)

    #Если эпизод завершен, сброс среды
    if done:
        print("Эпизод завершен!")
        observation, info = env.reset()

#Закрытие среды
env.close()

#Вывод информации о пространствах среды
print("\nИнформация о пространствах среды Taxi-v3:")
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")
print(f"Количество состояний: {env.observation_space.n}")
print(f"Количество действий: {env.action_space.n}")