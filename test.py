import requests
import time
import multiprocessing


def test():
    start = time.time()
    response = requests.post("http://localhost:10003/api/prompt_expansion", json={"prompt": "a cat"})
    assert response.status_code == 200
    print(response.json())
    print(f"Time: {time.time() - start}")

def main():
    N = 20
    # run in multiple processes
    processes = []
    for _ in range(N):
        p = multiprocessing.Process(target=test)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

main()