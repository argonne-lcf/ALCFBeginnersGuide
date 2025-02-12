# Early User Notes

The stability of the Aurora system has improved significantly over the months leading up to production availability, but its important to remember this is the first year of production. Unfortunately, stability issues remain (as it typical) and we encourage users to be proactive monitoring their jobs for expected behavior, reporting issues, and working with us on solutions.

## Outages and downtime

The current plan for Aurora includes weekly maintenance, typically on Mondays. There may be times where maintenance is canceled or delayed and we encourage users to watch for announcements in the aurora-notify list. Unfortunately, there may also be unplanned outages and downtime as well. We expect the frequency of downtime to decrease as the system stabilizes.

## Scheduling

Similar to other production ALCF systems, Aurora has a tiered scheduling policy whereby larger jobs can request longer maximum walltimes. To help ensure a positive user experience, users and projects are initially restricted to <=2048 nodes until they can demonstrate successful running of their workloads. We very much want to enable teams to leverage the full system to achieve their science goals.

## Storage

Aurora currently has separate home (Gecko) and project (flare) filesystems today. The plan is to cross-mount the other ALCF filesystems in the near future (possibly May 2025).

## Checkpointing

Checkpoint is essential. We strongly encourage users to be mindful of potential instabilities or other issues and to regularly checkpoint to minimize the impact from such disruptions. This will be particularly important as the scale of runs increase. 

## Getting help to resolve issues

This (Early User Notes and Known Issues)[https://docs.alcf.anl.gov/aurora/known-issues/] page will be updated regularly as Aurora stabilizes.

There are multiple avenues for getting assistance as users encounter issues. Users are encouraged to discuss issues with their ALCF points-of-contact (e.g. Catalyst team supporting INCITE and ALCC projects). 

The MyALCF (portal)[https://my.alcf.anl.gov/#/dashboard] provides convenient access to project and allocation information, training resources, and user guides.

Users are encourage to report isseus to (ALCF Support)[mailto:support@alcf.anl.gov]. It is important to provide job IDs, if you encounter issues while running on the system. Note, a support ticket may need to be created even after discussing with ALCF point-of-contact depending on the nature of the issue.

