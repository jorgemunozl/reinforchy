import time
from pynput import keyboard
import gymnasium as gym

# LunarLander actions: 0=do nothing, 1=left engine, 2=main engine, 3=right engine
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_MAIN = 2
ACTION_RIGHT = 3

pressed = set()

def on_press(key):
    try:
        pressed.add(key.char)
    except Exception:
        pressed.add(key)

def on_release(key):
    try:
        pressed.discard(key.char)
    except Exception:
        pressed.discard(key)
    if key == keyboard.Key.esc:
        return False  # stop listener

def get_action():
    # arrows or WASD
    if keyboard.Key.up in pressed or "w" in pressed:
        return ACTION_MAIN
    if keyboard.Key.left in pressed or "a" in pressed:
        return ACTION_LEFT
    if keyboard.Key.right in pressed or "d" in pressed:
        return ACTION_RIGHT
    return ACTION_NONE

env = gym.make("LunarLander-v3", render_mode="human")
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Controls: Up/W=main, Left/A=left, Right/D=right, ESC=quit")
obs, info = env.reset(seed=0)
terminated = truncated = False
ep_reward = 0.0

try:
    while True:
        action = get_action()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)

        if terminated or truncated:
            print(f"Episode reward: {ep_reward:.2f} (resetting)")
            obs, info = env.reset()
            terminated = truncated = False
            ep_reward = 0.0

        time.sleep(1 / 60)  # ~60 FPS
finally:
    env.close()
    listener.stop()