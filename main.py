
"""
Single-file RL client combining API operations, Q-learning agent, and an automatic game runner.
Uses two endpoints:
  - GW_ENDPOINT for world operations (locate, enter, move)
  - INDEX_ENDPOINT for runs and score
Moves use cardinal directions: N, S, W, E

cURL example to fetch last run:
  curl -s -X POST "${INDEX_ENDPOINT}" \
    -H "X-User-ID: ${USER_ID}" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"type":"runs","teamId":"${TEAM_ID}","count":1}' | jq .
"""
import os
import sys
import json
import random

# Ensure 'requests' is installed
try:
    import requests
except ImportError:
    print("Error: 'requests' library not installed. Run 'pip install requests'", file=sys.stderr)
    sys.exit(1)

# ——— Configuration —————————————————————————————————————————————————————
GW_ENDPOINT    = os.getenv(
    "GW_ENDPOINT",
    "https://www.notexponential.com/aip2pgaming/api/rl/gw.php"
)
INDEX_ENDPOINT = os.getenv(
    "INDEX_ENDPOINT",
    "https://www.notexponential.com/aip2pgaming/api/index.php"
)
TEAM_ID        = os.getenv("TEAM_ID", "1449")
USER_ID        = os.getenv("USER_ID", "3669")
API_KEY        = os.getenv("API_KEY", "60dea00b9a42f4329cdf")
DATA_FILE      = "points.json"

# ——— API Client —————————————————————————————————————————————————————
class APIClient:
    def __init__(self):
        self.gw_url = GW_ENDPOINT.rstrip("/")
        self.index_url = INDEX_ENDPOINT.rstrip("/")
        self.headers = {
            "X-User-ID": USER_ID,
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        self.team_id = TEAM_ID
        self.data_file = DATA_FILE
        self._ensure_data_file()

    def _ensure_data_file(self):
        if not os.path.exists(self.data_file):
            with open(self.data_file, "w") as f:
                json.dump({}, f)

    def _load_data(self):
        with open(self.data_file, "r") as f:
            return json.load(f)

    def _save_data(self, data):
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    # Index.php operations (use POST)
    def get_runs(self, count=1):
        payload = {"type": "runs", "teamId": self.team_id, "count": count}
        resp = requests.post(self.index_url, headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_score(self):
        payload = {"type": "score", "teamId": self.team_id}
        resp = requests.post(self.index_url, headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    # GW.php operations
    def get_location(self):
        params = {"type": "location", "teamId": self.team_id}
        resp = requests.get(self.gw_url, headers=self.headers, params=params)
        resp.raise_for_status()
        return resp.json()

    def enter_world(self, world_id):
        payload = {"type": "enter", "teamId": self.team_id, "worldId": world_id}
        resp = requests.post(self.gw_url, headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    def make_move(self, world_id, move):
        payload = {"type": "move", "teamId": self.team_id,
                   "worldId": world_id, "move": move}
        resp = requests.post(self.gw_url, headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    # Combined status fetch
    def get_all_status(self, world_id: str):
        # 1. Last run
        runs_resp = self.get_runs(1)
        last_run = None
        if runs_resp.get("code") == "OK":
            runs_list = runs_resp.get("runs", [])
            last_run = runs_list[-1] if runs_list else None
        # 2. Current location
        loc = self.get_location()
        # 3. Enter world if not in one
        if loc.get("worldId") == "-1":
            enter = self.enter_world(world_id)
        else:
            enter = {"currentRun": loc.get("runId")}
        return {"last_run": last_run, "location": loc, "enter": enter}

    # Local point persistence
    def store_points(self, world_id, points):
        data = self._load_data()
        team = data.setdefault(self.team_id, {"total": 0, "by_world": {}})
        team["by_world"][world_id] = points
        team["total"] = sum(team["by_world"].values())
        self._save_data(data)

    def load_points(self):
        data = self._load_data()
        return data.get(self.team_id, {"total": 0, "by_world": {}})

# ——— Q-Learning Agent —————————————————————————————————————————————————
class QLearningAgent:
    def __init__(
        self, client: APIClient, world_id: str,
        directions=None, alpha=0.1, gamma=0.9,
        epsilon=1.0, episodes=100,
        min_epsilon=0.01, decay_rate=0.995
    ):
        self.client = client
        self.world_id = world_id
        self.directions = directions or ["N", "S", "W", "E"]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.Q = {}

    def _state_key(self, state):
        return tuple(state) if isinstance(state, (list, tuple)) else (state,)

    def choose_direction(self, state_key, valid_dirs=None):
        dirs = valid_dirs or self.directions
        if random.random() < self.epsilon:
            return random.choice(dirs)
        q_vals = [self.Q.get((state_key, d), 0) for d in dirs]
        max_q = max(q_vals)
        best = [d for d, q in zip(dirs, q_vals) if q == max_q]
        return random.choice(best)

    def valid_directions(self, position, grid_size=40):
        x, y = position
        m = grid_size - 1
        dirs = []
        if x > 0: dirs.append("N")
        if x < m: dirs.append("S")
        if y > 0: dirs.append("W")
        if y < m: dirs.append("E")
        return dirs or self.directions

    def learn(self, state_key, action, reward, next_key, done):
        q_val = self.Q.get((state_key, action), 0)
        if done:
            target = reward
        else:
            future_max = max(self.Q.get((next_key, d), 0) for d in self.directions)
            target = reward + self.gamma * future_max
        self.Q[(state_key, action)] = q_val + self.alpha * (target - q_val)

    def train(self):
        best_reward = float('-inf')
        for _ in range(self.episodes):
            self.client.enter_world(self.world_id)
            loc = self.client.get_location()
            position = loc.get("position", [])
            state_key = self._state_key((loc.get("worldId"), tuple(position)))
            done = False
            total = 0
            while not done:
                valid = self.valid_directions(position)
                move = self.choose_direction(state_key, valid)
                res = self.client.make_move(self.world_id, move)
                reward = res.get("reward", 0)
                done = res.get("completed", False)
                position = res.get("position", [])
                next_key = self._state_key((res.get("worldId"), tuple(position)))
                self.learn(state_key, move, reward, next_key, done)
                state_key = next_key
                total += reward
            best_reward = max(best_reward, total)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        self.client.store_points(self.world_id, best_reward)
        return best_reward

# ——— Main —————————————————————————————————————————————————————————
def main():
    client = APIClient()

    status = client.get_all_status(input("Enter world ID: ").strip())
    print(json.dumps(status, indent=2))import os
import sys
import json
import random

# Ensure 'requests' is installed
try:
    import requests
except ImportError:
    print("Error: 'requests' library not installed. Run 'pip install requests'", file=sys.stderr)
    sys.exit(1)

# ——— Configuration ———————————————————————————————
GW_ENDPOINT = os.getenv("GW_ENDPOINT", "https://www.notexponential.com/aip2pgaming/api/rl/gw.php")
INDEX_ENDPOINT = os.getenv("INDEX_ENDPOINT", "https://www.notexponential.com/aip2pgaming/api/index.php")
TEAM_ID = os.getenv("TEAM_ID", "1449")
USER_ID = os.getenv("USER_ID", "3669")
API_KEY = os.getenv("API_KEY", "60dea00b9a42f4329cdf")
DATA_FILE = "points.json"

# ——— API Client ——————————————————————————————————————
class APIClient:
    def __init__(self):
        self.gw_url = GW_ENDPOINT.rstrip("/")
        self.index_url = INDEX_ENDPOINT.rstrip("/")
        self.base_headers = {
            "X-User-ID": USER_ID,
            "X-API-Key": API_KEY
        }
        self.team_id = TEAM_ID
        self.data_file = DATA_FILE
        self._ensure_data_file()

    def _ensure_data_file(self):
        if not os.path.exists(self.data_file):
            with open(self.data_file, "w") as f:
                json.dump({}, f)

    def _load_data(self):
        with open(self.data_file, "r") as f:
            return json.load(f)

    def _save_data(self, data):
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    # — Index.php operations (form-encoded) —
    def get_runs(self, count=1):
        payload = {"type": "runs", "teamId": self.team_id, "count": count}
        resp = requests.post(self.index_url, headers=self.base_headers, data=payload)
        resp.raise_for_status()
        return resp.json()

    def get_score(self):
        payload = {"type": "score", "teamId": self.team_id}
        resp = requests.post(self.index_url, headers=self.base_headers, data=payload)
        resp.raise_for_status()
        return resp.json()

    # — GW.php operations (JSON-based) —
    def get_location(self):
        params = {"type": "location", "teamId": self.team_id}
        resp = requests.get(self.gw_url, headers=self.base_headers, params=params)
        resp.raise_for_status()
        return resp.json()

    def enter_world(self, world_id):
        payload = {"type": "enter", "teamId": self.team_id, "worldId": world_id}
        headers = {**self.base_headers, "Content-Type": "application/json"}
        resp = requests.post(self.gw_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    def make_move(self, world_id, move):
        payload = {"type": "move", "teamId": self.team_id, "worldId": world_id, "move": move}
        headers = {**self.base_headers, "Content-Type": "application/json"}
        resp = requests.post(self.gw_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_all_status(self, world_id: str):
        runs_resp = self.get_runs(1)
        last_run = None
        if runs_resp.get("code") == "OK":
            runs_list = runs_resp.get("runs", [])
            last_run = runs_list[-1] if runs_list else None

        loc = self.get_location()

        if loc.get("worldId") == "-1":
            enter = self.enter_world(world_id)
        else:
            enter = {"currentRun": loc.get("runId")}

        return {"last_run": last_run, "location": loc, "enter": enter}

    def store_points(self, world_id, points):
        data = self._load_data()
        team = data.setdefault(self.team_id, {"total": 0, "by_world": {}})
        team["by_world"][world_id] = points
        team["total"] = sum(team["by_world"].values())
        self._save_data(data)

    def load_points(self):
        data = self._load_data()
        return data.get(self.team_id, {"total": 0, "by_world": {}})

# ——— Q-Learning Agent ——————————————————————————————
class QLearningAgent:
    def __init__(self, client: APIClient, world_id: str,
                 directions=None, alpha=0.1, gamma=0.9,
                 epsilon=1.0, episodes=100,
                 min_epsilon=0.01, decay_rate=0.995):
        self.client = client
        self.world_id = world_id
        self.directions = directions or ["N", "S", "W", "E"]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.Q = {}

    def _state_key(self, state):
        return tuple(state) if isinstance(state, (list, tuple)) else (state,)

    def choose_direction(self, state_key, valid_dirs=None):
        dirs = valid_dirs or self.directions
        if random.random() < self.epsilon:
            return random.choice(dirs)
        q_vals = [self.Q.get((state_key, d), 0) for d in dirs]
        max_q = max(q_vals)
        best = [d for d, q in zip(dirs, q_vals) if q == max_q]
        return random.choice(best)

    def valid_directions(self, position, grid_size=40):
        x, y = position
        m = grid_size - 1
        dirs = []
        if x > 0: dirs.append("N")
        if x < m: dirs.append("S")
        if y > 0: dirs.append("W")
        if y < m: dirs.append("E")
        return dirs or self.directions

    def learn(self, state_key, action, reward, next_key, done):
        q_val = self.Q.get((state_key, action), 0)
        if done:
            target = reward
        else:
            future_max = max(self.Q.get((next_key, d), 0) for d in self.directions)
            target = reward + self.gamma * future_max
        self.Q[(state_key, action)] = q_val + self.alpha * (target - q_val)

    def train(self):
        best_reward = float('-inf')
        for _ in range(self.episodes):
            self.client.enter_world(self.world_id)
            loc = self.client.get_location()
            position = loc.get("position", [])
            state_key = self._state_key((loc.get("worldId"), tuple(position)))
            done = False
            total = 0
            while not done:
                valid = self.valid_directions(position)
                move = self.choose_direction(state_key, valid)
                res = self.client.make_move(self.world_id, move)
                reward = res.get("reward", 0)
                done = res.get("completed", False)
                position = res.get("position", [])
                next_key = self._state_key((res.get("worldId"), tuple(position)))
                self.learn(state_key, move, reward, next_key, done)
                state_key = next_key
                total += reward
            best_reward = max(best_reward, total)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        self.client.store_points(self.world_id, best_reward)
        return best_reward

# ——— Main ——————————————————————————————————————————————
def main():
    client = APIClient()
    world_id = input("Enter world ID: ").strip()
    status = client.get_all_status(world_id)
    print(json.dumps(status, indent=2))

    agent = QLearningAgent(client, status.get("location", {}).get("worldId"))
    best = agent.train()
    print(f"Best total reward: {best}")

    print("Final score:", client.get_score())

if __name__ == "__main__":
    main()


    agent = QLearningAgent(client, status.get("location", {}).get("worldId"))
    best = agent.train()
    print(f"Best total reward: {best}")

    print("Final score:", client.get_score())

# Call main directly
main()
