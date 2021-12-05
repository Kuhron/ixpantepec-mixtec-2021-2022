# when WSL can't find /mnt/e
# https://superuser.com/questions/1360276/cannot-access-external-drive-in-windows-10-linux-subsystem-bash
sudo umount /mnt/e
sudo mount -t drvfs E: /mnt/e
