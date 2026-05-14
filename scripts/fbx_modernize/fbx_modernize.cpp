// Convert a legacy FBX file (e.g. FBX 3000) to FBX 2020 binary (version 7700).
//
// Build:  see Makefile in the same directory
// Usage:  ./fbx_modernize <input.fbx> <output.fbx>
//
// Background: the Ibaraki Radio Taiso mocap dataset ships as FBX 3000, which
// pre-dates everything Blender / pyassimp / current open-source tooling can
// read. AutoDesk's FBX SDK 2020.3.9 still understands FBX 3000 on the import
// side and writes modern FBX 7700 on the export side. See
// docs/fbx3000_intel_handoff.md for the full context.

#include <cstdio>
#include <cstring>
#include <fbxsdk.h>

static int find_fbx_binary_writer(FbxManager* mgr) {
    // The writer-ID we want is "FBX binary (*.fbx)". Don't trust format-ID 0
    // to be that — on some builds it's the ASCII writer.
    const int n = mgr->GetIOPluginRegistry()->GetWriterFormatCount();
    for (int i = 0; i < n; ++i) {
        const char* desc = mgr->GetIOPluginRegistry()->GetWriterFormatDescription(i);
        if (desc && std::strstr(desc, "FBX binary") != nullptr) return i;
    }
    return -1;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <input.fbx> <output.fbx>\n", argv[0]);
        return 2;
    }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];

    FbxManager* mgr = FbxManager::Create();
    FbxIOSettings* ios = FbxIOSettings::Create(mgr, IOSROOT);
    mgr->SetIOSettings(ios);
    ios->SetBoolProp(EXP_FBX_MATERIAL,        true);
    ios->SetBoolProp(EXP_FBX_TEXTURE,         true);
    ios->SetBoolProp(EXP_FBX_EMBEDDED,        false);
    ios->SetBoolProp(EXP_FBX_SHAPE,           true);
    ios->SetBoolProp(EXP_FBX_GOBO,            true);
    ios->SetBoolProp(EXP_FBX_ANIMATION,       true);
    ios->SetBoolProp(EXP_FBX_GLOBAL_SETTINGS, true);

    FbxImporter* importer = FbxImporter::Create(mgr, "");
    if (!importer->Initialize(in_path, -1, mgr->GetIOSettings())) {
        std::fprintf(stderr, "[fbx_modernize] importer init failed: %s\n",
                     importer->GetStatus().GetErrorString());
        importer->Destroy(); mgr->Destroy();
        return 3;
    }

    int v_major = 0, v_minor = 0, v_revision = 0;
    importer->GetFileVersion(v_major, v_minor, v_revision);
    std::printf("[fbx_modernize] input  %s  (FBX version %d.%d.%d)\n",
                in_path, v_major, v_minor, v_revision);

    FbxScene* scene = FbxScene::Create(mgr, "scene");
    if (!importer->Import(scene)) {
        std::fprintf(stderr, "[fbx_modernize] import failed: %s\n",
                     importer->GetStatus().GetErrorString());
        importer->Destroy(); mgr->Destroy();
        return 4;
    }
    importer->Destroy();

    const int writer_id = find_fbx_binary_writer(mgr);
    if (writer_id < 0) {
        std::fprintf(stderr, "[fbx_modernize] no FBX-binary writer registered\n");
        mgr->Destroy();
        return 5;
    }

    FbxExporter* exporter = FbxExporter::Create(mgr, "");
    if (!exporter->Initialize(out_path, writer_id, mgr->GetIOSettings())) {
        std::fprintf(stderr, "[fbx_modernize] exporter init failed: %s\n",
                     exporter->GetStatus().GetErrorString());
        exporter->Destroy(); mgr->Destroy();
        return 6;
    }
    // Pin to the highest binary version this SDK can write (FBX 2020 = 7700).
    exporter->SetFileExportVersion(FBX_2020_00_COMPATIBLE);

    if (!exporter->Export(scene)) {
        std::fprintf(stderr, "[fbx_modernize] export failed: %s\n",
                     exporter->GetStatus().GetErrorString());
        exporter->Destroy(); mgr->Destroy();
        return 7;
    }
    exporter->Destroy();
    mgr->Destroy();

    std::printf("[fbx_modernize] OK     %s\n", out_path);
    return 0;
}
