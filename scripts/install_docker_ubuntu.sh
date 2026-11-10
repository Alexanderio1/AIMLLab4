#!/usr/bin/env bash
# Official Docker Ubuntu repository. Does not remove conflicting packages.
set -euo pipefail
. /etc/os-release
if [ "$ID" != ubuntu ]; then echo 'This installer is for Ubuntu only'; exit 1; fi
arch=$(dpkg --print-architecture)
if [ "$arch" != arm64 ] && [ "$arch" != amd64 ]; then echo "Unsupported architecture: $arch"; exit 1; fi
sudo -v
if command -v docker >/dev/null && sudo docker version >/dev/null 2>&1 && sudo docker compose version >/dev/null 2>&1; then
  echo 'Docker Engine and Compose already work. Nothing installed.'
  sudo docker version
  sudo docker compose version
  exit 0
fi
for package in docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc; do
  status=$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)
  if [ "$status" = 'install ok installed' ]; then
    echo "Conflicting package found: $package. No packages removed. Send this output for diagnosis."
    exit 1
  fi
done
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
key=$(mktemp)
trap 'rm -f "$key"' EXIT
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o "$key"
sudo install -m 0644 "$key" /etc/apt/keyrings/docker.asc
sourcefile=$(mktemp)
cat > "$sourcefile" <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $arch
Signed-By: /etc/apt/keyrings/docker.asc
EOF
if [ -f /etc/apt/sources.list.d/docker.sources ] && ! cmp -s "$sourcefile" /etc/apt/sources.list.d/docker.sources; then
  rm -f "$sourcefile"
  echo 'Existing Docker repository configuration differs. Nothing overwritten. Send output for diagnosis.'
  exit 1
fi
sudo install -m 0644 "$sourcefile" /etc/apt/sources.list.d/docker.sources
rm -f "$sourcefile"
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
sudo docker compose version
