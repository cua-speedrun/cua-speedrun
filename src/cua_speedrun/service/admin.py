"""Operator commands for seasons, quotas, seeds, and invites."""

from __future__ import annotations

import argparse

from sqlalchemy import select

from cua_speedrun.service.db import Entry, Season, make_session_factory


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-seasons")

    freeze_p = sub.add_parser("freeze")
    freeze_p.add_argument("--season", required=True)

    tier_p = sub.add_parser("set-tier")
    tier_p.add_argument("--user", required=True, help="user handle")
    tier_p.add_argument("--tier", required=True, help="default | trusted | unlimited")

    sub.add_parser("seed-stats")

    inv_c = sub.add_parser("mint-invite", help="create an invite code")
    inv_c.add_argument("--note", default=None)
    inv_c.add_argument("--count", type=int, default=1)

    args = parser.parse_args()
    session_factory = make_session_factory()

    with session_factory() as session:
        if args.command == "list-seasons":
            keys = {e.season_key for e in session.execute(select(Entry)).scalars()}
            rows = {s.key: s for s in session.execute(select(Season)).scalars()}
            for key in sorted(keys | set(rows)):
                s = rows.get(key)
                status = s.status if s else "open"
                print(f"{status:<8} {key}")
        elif args.command == "freeze":
            from cua_speedrun.service.publishing import freeze_season

            freeze_season(session, args.season)
            print(f"frozen: {args.season}")
        elif args.command == "set-tier":
            from cua_speedrun.service.db import User

            user = session.execute(
                select(User).where(User.handle == args.user)
            ).scalar_one_or_none()
            if user is None:
                print(f"no such user: {args.user}")
            else:
                user.quota_tier = args.tier
                session.commit()
                print(f"{args.user}: tier = {args.tier}")
        elif args.command == "seed-stats":
            from cua_speedrun.service.db import BenchmarkRow
            from cua_speedrun.service.seeds import seed_pool_stats

            for b in session.execute(select(BenchmarkRow)).scalars():
                stats = seed_pool_stats(session, b.id)
                if stats:
                    print(f"{b.name}@{b.version}:")
                    for task, pools in stats.items():
                        sc = pools.get("scored", {})
                        pr = pools.get("practice", {})
                        print(f"  {task}: practice={pr.get('total', 0)} "
                              f"scored_retired={sc.get('used', 0)}")
        elif args.command == "mint-invite":
            from cua_speedrun.service.invites import mint

            for _ in range(args.count):
                print(mint(session, note=args.note))


if __name__ == "__main__":
    main()
