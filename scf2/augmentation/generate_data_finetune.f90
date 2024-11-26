
program test

    use libflit
    use librgm

    implicit none

    type(rgm2_curved) :: p
    integer :: i
    integer :: nt, nv, nm
    real, allocatable, dimension(:) :: rmo, fs, noise
    real, allocatable, dimension(:) :: slope, nsmooth1, nsmooth2
    integer, allocatable, dimension(:) :: nf, nl, pm
    real, allocatable, dimension(:, :) :: ppick, rmask, pf
    integer :: i1, i2, l1, l
    integer :: pick1, pick2
    integer :: dist
    integer, allocatable, dimension(:) :: nmeq
    integer, allocatable, dimension(:, :) :: blkrange
    real, allocatable, dimension(:) :: px, pz
    character(len=1024) :: dir_output
    real, allocatable, dimension(:, :) :: rp

    call mpistart

    call getpar_string('outdir', dir_output, './dataset_finetune')
    call getpar_int('ntrain', nt, 200)
    call getpar_int('nvalid', nv, 20)
    nm = nt + nv

    call make_directory(tidy(dir_output)//'/data_train')
    call make_directory(tidy(dir_output)//'/data_valid')
    call make_directory(tidy(dir_output)//'/target_train')
    call make_directory(tidy(dir_output)//'/target_valid')

    fs = random(nm, range=[160.0, 200.0])*0.9
    nl = nint(rescale(fs, range=[40.0, 60.0]))
    pm = random_order(nm)
    fs = fs(pm)
    nl = nl(pm)

    nf = irandom(nm, range=[15, 25])
    pm = random_order(nm)
    nf = nf(pm)

    noise = random(nm, range=[0.0, 0.1])
    nsmooth1 = random(nm, range=[0.0, 1.5])
    nsmooth2 = random(nm, range=[0.0, 1.5])
    where (nsmooth1 < 0.5)
        nsmooth1 = 0.0
    end where
    where (nsmooth2 < 0.5)
        nsmooth2 = 0.0
    end where

    slope = random(nm, range=[-25.0, 25.0])
    p%dip = [50.0, 130.0]

    nmeq = nint(rescale(nf*1.0, [400.0, 1000.0]))
    rmo = random(nm, range=[0.1, 0.6])

    call alloc_array(blkrange, [0, nrank - 1, 1, 2])
    call cut(1, nm, nrank, blkrange)

    do i = blkrange(rankid, 1), blkrange(rankid, 2)

        p%seed = i*i + 1000

        p%n1 = 256
        p%n2 = 1024
        p%yn_conv_noise = .false.
        p%nf = nf(i)
        p%f0 = fs(i)
        p%nl = nl(i)
        p%fwidth = 2.0
        p%refl_shape = 'gaussian'
        p%refl_mu2 = [0.5*p%n2,0.6*p%n2]
        p%refl_sigma2 = [200.0, 300.0]
        p%ng = 1
        p%refl_height = [0, 50]
        p%lwv = 0.1
        p%secondary_refl_height_ratio = 0.01

        p%refl_slope = slope(i)
        p%refl_smooth = 50
        p%refl_height = [0, 75]

        p%refl_slope_top = slope(i)/5.0
        p%refl_smooth_top = 20
        p%refl_height_top = [0.0, 5.0]

        p%lwv = 0.25
        p%wave = 'ricker'

        if (mod(irand(range=[1, nm]), 3) == 0) then
            p%unconf = irand(range=[1, 2])
            p%unconf_z = [0.01, 0.15]
            p%unconf_amp = [0.05, 0.1]
        else
            p%unconf = 0
        end if

        if (nf(i) > 28) then
            p%yn_regular_fault = .true.
            p%nf = nf(i)
            if (mod(irand(range=[1, 10]), 2) == 0) then
                p%dip = [rand(range=[100.0, 120.0]), rand(range=[60.0, 80.0])]
                p%disp = [4.0, -4.0]
            else
                p%dip = [rand(range=[60.0, 80.0]), rand(range=[100.0, 120.0])]
                p%disp = [-4.0, 4.0]
            end if
        else
            p%yn_regular_fault = .false.
            p%nf = nf(i)
            p%disp = [5.0, 10.0]
            p%dip = [50.0, 130.0]
            p%delta_dip = [5.0, 35.0]
        end if

        p%yn_fault = .true.
        p%noise_smooth = [nsmooth1(i), nsmooth2(i)]

        p%psf_sigma = [12.0, 8.0]
        p%noise_level = noise(i)

        call p%generate

        where (p%fault /= 0)
            p%fault = 1.0
        end where

        ! MEQ
        rmask = random_mask_smooth(p%n1, p%n2, gs=[4.0, 4.0], mask_out=rmo(i))
        pf = p%fault*rmask
        ppick = zeros(p%n1, p%n2)
        l1 = 0
        do l = 1, 5*maxval(nmeq)

            if (l1 < nmeq(i)) then

                if (p%yn_regular_fault) then
                    dist = irand(range=[0, 2])
                else
                    dist = irand(range=[0, 5])
                end if

                pick1 = irand(range=[dist + 1, p%n1 - dist])
                pick2 = irand(range=[dist + 1, p%n2 - dist])

                if (any(pf(pick1 - dist:pick1 + dist, pick2 - dist:pick2 + dist) == 1)) then
                    do i2 = -2*dist - 1, 2*dist + 1
                        do i1 = -2*dist - 1, 2*dist + 1
                            if (pick1 + i1 >= 1 .and. pick1 + i1 <= p%n1 &
                                    .and. pick2 + i2 >= 1 .and. pick2 + i2 <= p%n2) then
                                ppick(pick1 + i1, pick2 + i2) = &
                                    max(ppick(pick1 + i1, pick2 + i2), exp(-0.3*(i1**2 + i2**2)))
                            end if
                        end do
                    end do
                    l1 = l1 + 1
                end if

            end if

        end do


        ! Add seismicity noise
        dist = 1
        px = random(nint(0.05*nmeq(i)), range=[1, p%n1]*1.0)
        pz = random(nint(0.05*nmeq(i)), range=[1, p%n2]*1.0)

        do l = 1, size(px)
            pick1 = nint(px(l)) + 1
            pick2 = nint(pz(l)) + 1
            do i2 = -2*dist - 1, 2*dist + 1
                do i1 = -2*dist - 1, 2*dist + 1
                    if (pick1 + i1 >= 1 .and. pick1 + i1 <= p%n1 &
                            .and. pick2 + i2 >= 1 .and. pick2 + i2 <= p%n2) then
                        ppick(pick1 + i1, pick2 + i2) = &
                            max(ppick(pick1 + i1, pick2 + i2), exp(-0.3*(i1**2 + i2**2)))
                    end if
                end do
            end do
        end do

        rp = random(p%n1, p%n2)
        rp = gauss_filt(rp, [1, 1]*rand(range=[3.0, 10.0]))
        rp = rescale(rp, [0.3333, 1.0])
        p%image = p%image*rp
        p%image = p%image/std(p%image)

        if (i <= nt) then
            call output_array(p%image, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img.bin')
            call output_array(ppick, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_meq.bin')

            call output_array(p%fault*rmask, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0*rmask, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_fdip.bin')

            call output_array(p%fault, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fdip.bin')

        else
            call output_array(p%image, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img.bin')
            call output_array(ppick, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_meq.bin')
            call output_array(p%fault*rmask, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0*rmask, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_fdip.bin')

            call output_array(p%fault, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fdip.bin')
        end if

        if (i <= nt) then
            print *, date_time_compact(), ' train', i - 1, l1
        else
            print *, date_time_compact(), ' valid', i - nt - 1, l1
        end if

    end do

    call mpibarrier
    call mpiend

end program test
