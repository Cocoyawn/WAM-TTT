import datetime
print("now =", datetime.datetime.utcnow().strftime("%H:%M UTC"))
tot = {"libero_spatial":1627,"libero_object":1761,"libero_goal":1756,"libero_10":1824}
D = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/logs"
for suite, t in tot.items():
    done = 0
    for s in range(8):
        p = f"{D}/plus_envgen_attnmix_5dim_{suite}_shard{s}.log"
        try:
            done += open(p, errors="ignore").read().count("Current task success rate:")
        except FileNotFoundError:
            pass
    print(f"  {suite:<14} done={done:<5} / {t}   ({100*done/t:.1f}%)")
