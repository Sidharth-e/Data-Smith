const fs = require("fs");
const path = require("path");

const dirsToSearch = [
  "/Users/sidharthe/Library/pnpm/store",
  "/Users/sidharthe/.npm",
  "/Users/sidharthe/.cache",
];

function searchDir(dir, targetName) {
  try {
    if (!fs.existsSync(dir)) return null;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === targetName) {
          const pkgJson = path.join(fullPath, "package.json");
          if (fs.existsSync(pkgJson)) {
            return fullPath;
          }
        }
        const res = searchDir(fullPath, targetName);
        if (res) return res;
      }
    }
  } catch (e) {}
  return null;
}

console.log("Searching for lucide-react...");
for (const root of dirsToSearch) {
  const found = searchDir(root, "lucide-react");
  if (found) {
    console.log("Found lucide-react at:", found);
  }
}

console.log("Searching for zustand...");
for (const root of dirsToSearch) {
  const found = searchDir(root, "zustand");
  if (found) {
    console.log("Found zustand at:", found);
  }
}
