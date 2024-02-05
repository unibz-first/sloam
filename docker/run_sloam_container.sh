SLOAMWS="/home/mchang/sloam_ws"
# BAGS_DIR='/media/gnardari/DATA/bags/sloam'
BAGS_DIR='/home/mchang/Downloads/bags/sloam_bags'
trap "rm -f /tmp/.docker.xauth" EXIT # for the non-reckless, see line 25

# docker run -it \
#     --name="sloam_ros" \
#     --net="host" \
#     --privileged \
#     --env="DISPLAY=$DISPLAY" \
#     --env="QT_X11_NO_MITSHM=1" \
#     --rm \
#     --workdir="/opt/sloam_ws" \
#     --volume="$SLOAMWS:/opt/sloam_ws" \
#     --volume="$HOME:/root" \
#     --volume="/home/$USER/repos:/home/$USER/repos" \
#     --volume="/home/$USER/repos:/home/$USER/repos" \
#     --volume="/home/$USER/ros:/home/$USER/ros" \
#     --volume="$BAGS_DIR:/opt/bags" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     gnardari/sloam:latest \
#     bash

# xhost +local:root # for the lazy and reckless

#    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    #--env="XAUTHORITY=$XAUTH" \
# less reckless but still: thanks github copilot: 
touch /tmp/.docker.xauth # creates tmp key to terminate post-run
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
# lists xauth entries | mod xauth docker IPs to wildcard | merge mod entries to new file
sudo docker run -it \
    --name="sloam_ros" \
    --net="host" \
    --privileged \
    --rm \
    --gpus="all" \
    --workdir="/opt/sloam_ws" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=/tmp/.docker.xauth" \
    --volume="$SLOAMWS:/opt/sloam_ws" \
    --volume="$BAGS_DIR:/opt/bags" \
    --volume="/tmp/.docker.xauth:/tmp/.docker.xauth:rw" \
    --volume="/home/$USER/repos:/home/$USER/repos" \
    gnardari/sloam:runtime \
    bash

rm /tmp/.docker.xauth # double tap
