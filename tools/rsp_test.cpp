// Runs RSP memory images on the low-level RSP (src/rsp_core.*) and reports
// how they ended. An image is 4 KB of IMEM followed by 4 KB of DMEM, as
// ORBIT64_RSP_DUMP=<dir> saves one for every task the low-level RSP starts.
// The N64 diagnostics cartridge's RSP sub-tests store 0xFEED (pass) or
// 0xDEADBEEF (fail) at DMEM 0xF84 before their BREAK, with the number of
// the check in r30.
//
//   make rsp_test && bin/rsp_test dump/*.bin

#include "mi.hpp"
#include "rdp.hpp"
#include "rsp.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    int failed = 0;
    std::vector<u8> rdram(RDRAM_SIZE);
    for (int i = 1; i < argc; ++i) {
        FILE* f = std::fopen(argv[i], "rb");
        if (!f) { std::fprintf(stderr, "%s: can't open\n", argv[i]); return 1; }
        MI mi;
        RDP rdp;
        RSP rsp;
        std::fread(rsp.get_imem(), 1, 4096, f);
        std::fread(rsp.get_dmem(), 1, 4096, f);
        std::fclose(f);
        rsp.set_force_lle(true);
        rsp.write_reg(0x04080000, 0, mi, rdp, rdram.data(), rdram.size());
        rsp.write_reg(0x04040010, 1, mi, rdp, rdram.data(), rdram.size()); // clear halt
        u64 cycles = 0;
        while (!(rsp.get_status() & SPStatus::HALT) && cycles < 50'000'000) {
            rsp.step(3000, mi, rdp, rdram.data(), rdram.size());
            cycles += 2000;
        }
        const u8* d = rsp.get_dmem();
        const u32 result = (u32(d[0xF84]) << 24) | (u32(d[0xF85]) << 16) | (u32(d[0xF86]) << 8) | d[0xF87];
        const char* verdict = result == 0xFEED0000 ? "pass" : result == 0xDEADBEEF ? "FAIL" : "?";
        if (result != 0xFEED0000) ++failed;
        std::printf("%-40s %s  r30=%u pc=%03x status=%x\n", argv[i], verdict, rsp.debug_core().gpr(30),
                    rsp.debug_core().pc, rsp.get_status());
        if (std::getenv("RSP_TEST_VREGS"))
            for (int v = 0; v < 32; ++v) {
                std::printf("  v%-2d", v);
                for (int e = 0; e < 8; ++e) std::printf(" %04x", rsp.debug_core().vreg(v).e[e]);
                std::printf("\n");
            }
    }
    return failed ? 1 : 0;
}
